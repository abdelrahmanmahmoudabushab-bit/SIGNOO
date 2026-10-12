"""Keeping held-out signs out of the training manifest."""
import json

import pytest

import keep_signs


def write_manifest(tmp_path):
    records = [{"signer_id": signer, "sign_id": sign, "features": f"{signer}/{sign}/0.npy"}
               for signer in ("s1", "s2") for sign in ("dog", "cat", "book")]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path, records


def test_only_the_listed_signs_remain_and_their_records_are_unchanged(tmp_path):
    _, records = write_manifest(tmp_path)

    kept = keep_signs.keep_signs(records, {"dog", "cat"})

    assert {r["sign_id"] for r in kept} == {"dog", "cat"}
    assert kept == [r for r in records if r["sign_id"] != "book"]


def test_the_script_writes_next_to_the_input_so_feature_paths_stay_valid(tmp_path, capsys):
    manifest, _ = write_manifest(tmp_path)

    keep_signs.main([str(manifest), "--keep", "dog", "cat"])

    out = json.loads((tmp_path / "manifest_known.json").read_text(encoding="utf-8"))
    assert len(out) == 4 and "book" not in {r["sign_id"] for r in out}
    assert "dropped: book" in capsys.readouterr().out
    assert len(json.loads(manifest.read_text(encoding="utf-8"))) == 6      # the full manifest is left alone


def test_a_misspelt_sign_is_an_error_not_a_silently_smaller_vocabulary(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    with pytest.raises(SystemExit, match="dgo"):
        keep_signs.main([str(manifest), "--keep", "dgo", "cat"])
    assert not (tmp_path / "manifest_known.json").exists()


def test_an_output_in_another_directory_is_refused(tmp_path):
    manifest, _ = write_manifest(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(SystemExit, match="relative"):
        keep_signs.main([str(manifest), "--keep", "dog", "--out", str(tmp_path / "elsewhere" / "m.json")])
