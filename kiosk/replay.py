"""Replays recorded videos through the kiosk's real B2 -> B3 -> B4 path (NFR-N1).

    python -m kiosk.replay data/popsign/raw --language asl --signers 7003 7007

Each frame is resized and JPEG-encoded the way the browser sends it, paced at the
video's frame rate, and fed to the same live-stream perception, pause-based
segmenter and per-language recognizer the kiosk uses. That is what separates this
from `ml/evaluate.py`, which scores pre-extracted features and so cannot see the
failures that only exist live: a word the segmenter never closes, one it splits in
two, or perception losing the hand mid-sign.

Videos are expected as `<root>/<signer>/<sign>/<clip>.mp4`; the folder names are the
ground truth.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

from kiosk.config import CONFIG, REPO_ROOT, SEGMENTATION_DEFAULTS
from kiosk.languages import LanguagePack, LanguageRegistry
from kiosk.perception import HolisticPerception

MAX_SIDE = 640          # what a browser canvas sends, not the camera's native size
JPEG_QUALITY = 70
TAIL_SECONDS = 1.5      # still frames after the clip, so a final pause can close the word


def replay_video(path: Path, pack: LanguagePack, realtime: bool = True,
                 tracker_fps: float | None = None) -> tuple[list, dict]:
    """Returns (one Hypothesis per utterance the segmenter emitted, diagnostics).

    The diagnostics say how many perception results arrived, how many showed a hand,
    and how many frames each utterance held, which is what tells a tracking failure
    from a segmentation failure.

    `tracker_fps` gives the tracker only that many frames a second, each stamped with
    its real time in the video. A development machine tracks at the video's full rate;
    the Jetson's CPU reaches about 10 a second (measured), and whether a word survives
    that is exactly what a full-rate replay cannot show.
    """
    import cv2

    # The language's own boundary tuning over the kiosk defaults, as kiosk/main.py does.
    segmenter = pack.make_segmenter(SEGMENTATION_DEFAULTS)
    recognizer = pack.recognizer
    perception = HolisticPerception(stride=CONFIG.perception_stride)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hypotheses = []
    stats = {"results": 0, "hand_frames": 0, "utterance_frames": []}
    last_jpeg = None
    index = 0          # position in the video, counting every frame, fed or skipped
    next_due_s = 0.0   # video time at which the tracker is next given a frame

    def feed(jpeg: bytes) -> None:
        nonlocal index, next_due_s
        video_time_s = index / fps
        index += 1
        if tracker_fps:
            if video_time_s < next_due_s:
                return
            next_due_s = max(next_due_s + 1.0 / tracker_fps, video_time_s)
        perception.submit_frame(jpeg, int(video_time_s * 1000))
        landmarks = perception.latest()
        if landmarks is None:
            return
        stats["results"] += 1
        stats["hand_frames"] += landmarks.left is not None or landmarks.right is not None
        utterance = segmenter.feed(landmarks)
        if utterance is not None:
            stats["utterance_frames"].append(len(utterance.landmarks))
            hypotheses.append(recognizer.classify(utterance))

    try:
        while True:
            started = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                break
            scale = MAX_SIDE / max(frame.shape[:2])
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            last_jpeg = buf.tobytes()
            feed(last_jpeg)
            if realtime:
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))
        for _ in range(int(TAIL_SECONDS * fps)):
            if last_jpeg is None:
                break
            feed(last_jpeg)
            if realtime:
                time.sleep(1.0 / fps)
    finally:
        cap.release()
        perception.close()
    return hypotheses, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="folder of <signer>/<sign>/<clip>.mp4")
    ap.add_argument("--language", default="asl")
    ap.add_argument("--signers", nargs="*", help="only these signer folders (default: all)")
    ap.add_argument("--signs", nargs="*", help="only these sign folders (default: all)")
    ap.add_argument("--model", help="checkpoint to test instead of the language's configured one")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips")
    ap.add_argument("--fast", action="store_true", help="do not pace to the video's frame rate")
    ap.add_argument("--tracker-fps", type=float, default=0,
                    help="give the tracker only this many frames a second (the Jetson reaches about 10)")
    args = ap.parse_args()

    registry = LanguageRegistry(CONFIG.languages_path, REPO_ROOT, CONFIG.reject_threshold,
                                CONFIG.confident_threshold, CONFIG.top_k_alternatives,
                                overrides={args.language: {"model": Path(args.model)}} if args.model else None)
    pack = registry.get(args.language)
    if pack is None or not pack.ready:
        raise SystemExit(f"language {args.language!r} has no loadable model")

    videos = sorted(Path(args.root).rglob("*.mp4"))
    if args.signers:
        videos = [v for v in videos if v.parent.parent.name in set(args.signers)]
    if args.signs:
        videos = [v for v in videos if v.parent.name in set(args.signs)]
    if args.limit:
        videos = videos[: args.limit]

    recognizer = pack.recognizer
    known_signs = set(recognizer.labels)
    outcome = {"known": Counter(), "unknown": Counter()}
    totals = Counter()
    harmful = []
    for n, video in enumerate(videos, 1):
        truth = video.parent.name
        group = "known" if truth in known_signs else "unknown"
        totals[group] += 1
        hyps, stats = replay_video(video, pack, realtime=not args.fast, tracker_fps=args.tracker_fps or None)

        # What the kiosk would do with each utterance: commit a word, offer candidates
        # for the visitor to confirm, or refuse. Judge the clip by its most confident one.
        def status(h) -> str:
            if h.gloss is None:
                return "rejected"
            return "committed" if h.confidence >= recognizer.commit_bar(h.gloss) else "candidates"

        best = max(hyps, key=lambda h: h.confidence, default=None)
        if best is None:
            result = "no_utterance"
        else:
            result = status(best)
            if result == "committed":
                result = "committed_correct" if best.gloss == truth else "committed_WRONG"
                if result == "committed_WRONG":
                    harmful.append(f"{truth} -> {best.gloss} ({best.confidence:.2f})  {video.name}")
            elif result == "candidates" and truth in [g for g, _ in best.alternatives]:
                result = "candidates_with_truth"
        outcome[group][result] += 1
        if sum(status(h) == "committed" for h in hyps) > 1:
            outcome[group]["extra_committed_words"] += 1
        print(f"[{n}/{len(videos)}] {truth:9s} utterances={len(hyps)} "
              f"{result}  said={best.gloss if best else None}  results={stats['results']} "
              f"hand_frames={stats['hand_frames']} utt_frames={stats['utterance_frames']} "
              f"{video.parent.parent.name}", flush=True)

    rate = f"{args.tracker_fps:g} tracked frames/s" if args.tracker_fps else "every frame tracked"
    print(f"\nlive path ({args.language}, {rate}); commits need confidence >= {recognizer.confident_threshold}")
    for group, keys in (
        ("known", ("committed_correct", "committed_WRONG", "candidates_with_truth", "candidates",
                   "rejected", "no_utterance", "extra_committed_words")),
        ("unknown", ("committed_WRONG", "committed_correct", "candidates", "candidates_with_truth",
                     "rejected", "no_utterance", "extra_committed_words")),
    ):
        total = totals[group]
        if not total:
            continue
        print(f"\n{group.upper()} signs: {total} clips")
        for key in keys:
            if outcome[group][key]:
                label = key if group == "known" or key != "committed_WRONG" else "committed (false accept)"
                print(f"  {label:26s} {outcome[group][key]:4d}  ({outcome[group][key] / total:.0%})")
    for line in harmful[:15]:
        print("  confident mistake:", line)


if __name__ == "__main__":
    main()
