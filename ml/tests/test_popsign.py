"""PopSign fetch/unpack, without a network or real videos: fake tars, a fake server."""
import email.message
import io
import tarfile
import urllib.error

import pytest

import popsign

SHORT = "4a.7005-dog-2023_01_03_15_08_59.922-0.mp4"
LONG = "gtsignstudy4a.8045-dog-2023_01_27_12_18_58.117-0.mp4"


def clip(signer, n, prefix="4a"):
    return f"{prefix}.{signer}-dog-2023_01_0{n}_10_00_00.000-0.mp4"


# ── signer id ─────────────────────────────────────────────────────────────────

def test_signer_id_is_read_from_both_file_name_forms():
    assert popsign.signer_id(SHORT) == "7005"
    assert popsign.signer_id(LONG) == "8045"


def test_a_name_without_a_signer_id_gives_none():
    assert popsign.signer_id("dog-2023_01_03_15_08_59.922-0.mp4") is None


# ── per-signer selection ──────────────────────────────────────────────────────

def test_select_caps_each_signer_and_still_takes_every_signer():
    names = [clip("7005", n) for n in range(1, 8)] + [clip("8045", 1)] + [clip("9001", n) for n in (1, 2)]
    chosen = popsign.select(names, existing=[], per_signer=3)

    per_signer = {s: sum(popsign.signer_id(n) == s for n in chosen) for s in ("7005", "8045", "9001")}
    assert per_signer == {"7005": 3, "8045": 1, "9001": 2}      # a prolific signer cannot crowd out the rest


def test_select_is_idempotent_and_counts_what_is_already_unpacked():
    names = [clip("7005", n) for n in range(1, 8)]
    first = popsign.select(names, existing=[], per_signer=3)

    assert popsign.select(names, existing=first, per_signer=3) == []
    assert len(popsign.select(names, existing=first[:2], per_signer=3)) == 1
    # Same answer whatever order the archives list the clips in.
    assert popsign.select(reversed(names), existing=[], per_signer=3) == first


def test_select_leaves_out_clips_with_no_signer_id():
    assert popsign.select(["dog-nobody.mp4", clip("7005", 1)], [], 3) == [clip("7005", 1)]


# ── unpack ────────────────────────────────────────────────────────────────────

def make_tar(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))


def fake_archives(tars):
    """Signer 7005 has five clips spread over the three splits, 8045 has two."""
    make_tar(tars / "train" / "dog.tar", {
        f"dog/{clip('7005', 1)}": b"a1", f"dog/{clip('7005', 2)}": b"a2",
        f"dog/{clip('8045', 1, 'gtsignstudy4a')}": b"b1",
        "dog/notes.txt": b"not a video", "dog/dog-nobody.mp4": b"no signer id",
    })
    make_tar(tars / "val" / "dog.tar", {f"dog/{clip('7005', 3)}": b"a3", f"dog/{clip('7005', 4)}": b"a4"})
    make_tar(tars / "test" / "dog.tar", {
        f"dog/{clip('7005', 5)}": b"a5", f"dog/{clip('8045', 2, 'gtsignstudy4a')}": b"b2",
    })


def test_unpack_builds_the_signer_sign_layout_capped_per_signer(tmp_path):
    fake_archives(tmp_path / "tars")
    raw = tmp_path / "raw"

    result = popsign.unpack_sign("dog", tmp_path / "tars", raw, per_signer=3)

    assert (result["added"], result["total"], result["signers"]) == (5, 5, 2)
    assert result["no_signer_id"] == 1
    # The cap is across splits: 7005's three come from train and val, none from test.
    assert sorted(p.name for p in (raw / "7005" / "dog").iterdir()) == [clip("7005", n) for n in (1, 2, 3)]
    assert len(list((raw / "8045" / "dog").iterdir())) == 2
    assert (raw / "7005" / "dog" / clip("7005", 3)).read_bytes() == b"a3"
    assert sorted(p.name for p in raw.iterdir()) == ["7005", "8045"]      # nothing for the id-less clip


def test_unpack_twice_adds_nothing_and_never_exceeds_the_cap(tmp_path):
    fake_archives(tmp_path / "tars")
    raw = tmp_path / "raw"
    popsign.unpack_sign("dog", tmp_path / "tars", raw, per_signer=3)

    again = popsign.unpack_sign("dog", tmp_path / "tars", raw, per_signer=3)

    assert (again["added"], again["total"]) == (0, 5)
    assert len(list((raw / "7005" / "dog").iterdir())) == 3


def test_unpack_prints_one_summary_line_per_sign_and_fails_on_a_sign_with_no_archive(tmp_path, capsys):
    fake_archives(tmp_path / "tars")

    failed = popsign.unpack(["dog", "cat"], tmp_path / "tars", tmp_path / "raw", per_signer=3)

    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0].startswith("dog: +5 clips, 5 total, 2 signers")
    assert lines[1].startswith("cat: no archives")
    assert failed == 1


# ── fetch ─────────────────────────────────────────────────────────────────────

def tar_bytes(payload=b"x" * 5000):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("dog/clip.mp4")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class Response(io.BytesIO):
    """The parts of urlopen's response that download() uses."""

    def __init__(self, body, status=200, length=None, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body) if length is None else length)}
        if headers is not None:
            self.headers = headers


def http_error(code, headers=None):
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError("http://x/dog.tar", code, "error", message, None)


def test_a_404_is_reported_and_nothing_is_written(tmp_path):
    def opener(request, timeout):
        raise http_error(404)

    status, _ = popsign.download("http://x/tv.tar", tmp_path / "train" / "tv.tar", opener)

    assert status == "missing"
    assert list((tmp_path / "train").iterdir()) == []


def test_an_error_page_served_with_status_200_is_not_kept_as_a_tar(tmp_path):
    def opener(request, timeout):
        return Response(b"<html>Not Found</html>")

    status, _ = popsign.download("http://x/dog.tar", tmp_path / "dog.tar", opener)

    assert status.startswith("failed")
    assert list(tmp_path.iterdir()) == []


def test_a_partial_file_is_resumed_from_where_it_stopped(tmp_path):
    body = tar_bytes()
    dest = tmp_path / "dog.tar"
    (tmp_path / "dog.tar.part").write_bytes(body[:1000])
    asked = []

    def opener(request, timeout):
        asked.append(request.get_header("Range"))
        return Response(body[1000:], status=206)

    status, size = popsign.download("http://x/dog.tar", dest, opener)

    assert (status, size) == ("ok", len(body))
    assert asked == ["bytes=1000-"]
    assert dest.read_bytes() == body
    assert not (tmp_path / "dog.tar.part").exists()


def test_a_dropped_connection_is_retried_from_the_bytes_already_received(tmp_path):
    body = tar_bytes()
    dest = tmp_path / "dog.tar"
    asked = []

    def opener(request, timeout):
        asked.append(request.get_header("Range"))
        if len(asked) == 1:
            return Response(body[:2000], length=len(body))          # promises all, sends some
        return Response(body[2000:], status=206)

    status, _ = popsign.download("http://x/dog.tar", dest, opener, pause=0)

    assert status == "ok"
    assert asked == [None, "bytes=2000-"]
    assert dest.read_bytes() == body


def test_a_download_that_keeps_dropping_fails_and_leaves_no_tar(tmp_path):
    body = tar_bytes()

    def opener(request, timeout):
        return Response(b"", status=206, length=len(body))

    (tmp_path / "dog.tar.part").write_bytes(body[:1000])
    status, _ = popsign.download("http://x/dog.tar", tmp_path / "dog.tar", opener, retries=2, pause=0)

    assert status.startswith("failed")
    assert not (tmp_path / "dog.tar").exists()
    assert (tmp_path / "dog.tar.part").exists()                      # kept, so the next run resumes


def test_a_complete_archive_is_not_downloaded_again(tmp_path):
    dest = tmp_path / "dog.tar"
    dest.write_bytes(tar_bytes())

    def opener(request, timeout):
        raise AssertionError("should not have asked the server")

    assert popsign.download("http://x/dog.tar", dest, opener)[0] == "exists"


def test_a_cut_off_download_with_no_content_length_is_not_kept_as_a_tar(tmp_path):
    body = tar_bytes()

    def opener(request, timeout):
        return Response(body[:2000], headers={})         # first header intact, the clip cut short

    status, _ = popsign.download("http://x/dog.tar", tmp_path / "dog.tar", opener)

    assert status.startswith("failed")
    assert list(tmp_path.iterdir()) == []


def test_a_server_that_ignores_range_and_sends_everything_replaces_the_partial_file(tmp_path):
    body = tar_bytes()
    dest = tmp_path / "dog.tar"
    (tmp_path / "dog.tar.part").write_bytes(b"stale bytes from some other file")

    def opener(request, timeout):
        assert request.get_header("Range")
        return Response(body, status=200)

    assert popsign.download("http://x/dog.tar", dest, opener)[0] == "ok"
    assert dest.read_bytes() == body                     # not appended to the stale bytes


def test_a_416_on_a_partial_file_that_is_already_whole_finishes_it(tmp_path):
    body = tar_bytes()
    dest = tmp_path / "dog.tar"
    (tmp_path / "dog.tar.part").write_bytes(body)

    def opener(request, timeout):
        raise http_error(416, {"Content-Range": f"bytes */{len(body)}"})

    assert popsign.download("http://x/dog.tar", dest, opener)[0] == "ok"
    assert dest.read_bytes() == body


def test_a_416_on_a_partial_file_of_the_wrong_size_starts_over(tmp_path):
    body = tar_bytes()
    dest = tmp_path / "dog.tar"
    (tmp_path / "dog.tar.part").write_bytes(body + b"longer than the real file")
    asked = []

    def opener(request, timeout):
        asked.append(request.get_header("Range"))
        if len(asked) == 1:
            raise http_error(416, {"Content-Range": f"bytes */{len(body)}"})
        return Response(body)

    assert popsign.download("http://x/dog.tar", dest, opener, pause=0)[0] == "ok"
    assert asked[1] is None
    assert dest.read_bytes() == body


def test_busy_and_server_errors_are_retried_a_bounded_number_of_times(tmp_path):
    body = tar_bytes()
    codes = iter([429, 503])
    calls = []

    def flaky(request, timeout):
        calls.append(1)
        code = next(codes, None)
        if code:
            raise http_error(code)
        return Response(body)

    assert popsign.download("http://x/dog.tar", tmp_path / "dog.tar", flaky, pause=0)[0] == "ok"
    assert len(calls) == 3

    def down(request, timeout):
        calls.append(1)
        raise http_error(503)

    del calls[:]
    status, _ = popsign.download("http://x/cat.tar", tmp_path / "cat.tar", down, retries=3, pause=0)
    assert status.startswith("failed: HTTP 503") and len(calls) == 3
    assert not (tmp_path / "cat.tar").exists()


def test_a_403_is_not_retried(tmp_path):
    calls = []

    def opener(request, timeout):
        calls.append(1)
        raise http_error(403)

    assert popsign.download("http://x/dog.tar", tmp_path / "dog.tar", opener, pause=0)[0] == "failed: HTTP 403"
    assert len(calls) == 1


# ── hostile or broken input ───────────────────────────────────────────────────

def test_a_member_name_with_backslashes_cannot_write_outside_the_raw_root(tmp_path):
    tars, raw = tmp_path / "tars", tmp_path / "out" / "deep" / "raw"
    make_tar(tars / "train" / "dog.tar", {
        "dog/..\\..\\..\\4a.1-evil.mp4": b"escaped",
        "dog/..\\4a.2-evil.mp4": b"escaped",
        f"dog/{clip('7005', 1)}": b"fine",
    })

    result = popsign.unpack_sign("dog", tars, raw, per_signer=3)

    assert result["added"] == 1
    written = sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert written == [f"out/deep/raw/7005/dog/{clip('7005', 1)}"]


def test_a_corrupt_archive_is_reported_and_the_other_splits_are_still_unpacked(tmp_path, capsys):
    fake_archives(tmp_path / "tars")
    whole = (tmp_path / "tars" / "val" / "dog.tar").read_bytes()
    (tmp_path / "tars" / "val" / "dog.tar").write_bytes(whole[:700])       # cut inside the first clip
    (tmp_path / "tars" / "test" / "dog.tar").write_bytes(b"<html>Not Found</html>")

    failed = popsign.unpack(["dog"], tmp_path / "tars", tmp_path / "raw", per_signer=3)

    line = capsys.readouterr().out.strip()
    assert line.startswith("dog: +3 clips, 3 total, 2 signers")           # train's clips only
    assert "UNREADABLE archive for: val, test" in line
    assert failed == 1


def test_sign_names_that_are_not_plain_folder_names_are_refused(tmp_path):
    assert popsign.valid_sign("dog") and popsign.valid_sign("TV") and popsign.valid_sign("ice-cream_2")
    for bad in ("../dog", "a/b", "a\\b", "c:dog", "..", "dog.tar", "*", ""):
        assert not popsign.valid_sign(bad), bad

    with pytest.raises(SystemExit, match="plain sign names"):
        popsign.main(["unpack", "../dog", "--tars", str(tmp_path), "--out", str(tmp_path / "raw")])
    with pytest.raises(SystemExit, match="plain sign names"):
        popsign.main(["fetch", "dog", "a/b", "--out", str(tmp_path / "tars")])
    assert list(tmp_path.iterdir()) == []
