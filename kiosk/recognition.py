"""B4 Recognition: utterance -> gloss hypothesis with confidence (SYSTEM_DESIGN.md B4).

Loads the checkpoint built per `models/README.md` (currently the synthetic-data
model — see that file's caveat) and reuses `ml/model.py`'s `SignTransformer`
unchanged, and `core/features.py` for the exact same feature encoding `ml/train.py`
trained on. Importing anything else from `ml/` here would risk skew; importing
`core/features.py` is the whole point of `core/` existing (SYSTEM_DESIGN.md §7).

Rejection is a first-class output (Requirements §6.3): below `reject_threshold`,
`Hypothesis.gloss` is `None` rather than a guess.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from core.contracts import Hypothesis, Utterance
from core.features import FRAME_DIM, INPUT_DIMS, SEQ_LEN, model_input, normalise_frame, resample

# Feature versions B2 Perception supplies everything for. Version 3 also needs face landmarks
# (core.features.face_features), which the live pipeline does not carry yet, so a version-3
# checkpoint is refused here instead of being run on features it was never trained with.
LIVE_FEATURE_VERSIONS = (1, 2, 4, 5)

_ML_DIR = Path(__file__).resolve().parent.parent / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))


# A model trained with this label learns "none of my signs". Answering it is a refusal,
# which is how a closed vocabulary avoids naming a known sign for something it was
# never taught. Built by ml/make_open_set.py.
OTHER_LABEL = "__other__"


class ModelUnavailable(RuntimeError):
    """Raised when the checkpoint is missing or corrupt (SYSTEM_DESIGN.md §6:
    "Model file corrupt -> refuse to start with that model"). The caller decides
    whether that means refusing to start, or serving a clear out-of-service state.
    """


class Recognizer:
    def __init__(self, model_path: Path, reject_threshold: float, confident_threshold: float,
                 top_k: int):
        self.reject_threshold = reject_threshold
        self.confident_threshold = confident_threshold
        self.top_k = top_k
        self.model_path = model_path
        self._load()

    def _load(self) -> None:
        from model import build_model  # ml/model.py, added to sys.path above

        if not self.model_path.exists():
            raise ModelUnavailable(
                f"checkpoint not found at {self.model_path} — build one first, see "
                "models/README.md (python ml/make_synthetic.py && python ml/train.py ...)"
            )
        try:
            # weights_only: a checkpoint is data (tensors, labels, ints). Refusing to
            # unpickle arbitrary objects means a tampered model file cannot run code
            # on the kiosk when the server rolls models out.
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except Exception as exc:  # corrupt file, incompatible torch version, etc.
            raise ModelUnavailable(f"checkpoint at {self.model_path} could not be loaded: {exc}") from exc

        required = {"labels", "input_dim"} | ({"members"} if "members" in checkpoint else {"model"})
        missing = required - checkpoint.keys()
        if missing:
            raise ModelUnavailable(
                f"checkpoint at {self.model_path} is missing keys {missing} — refusing to load "
                "a model whose shape we cannot trust"
            )

        # ml/train.py saves `labels` as the {sign_id: class_index} mapping built by
        # ml/data.py's label_index() — invert it once here so classify() can index
        # straight into a list by the model's output class index.
        label_to_index: dict[str, int] = checkpoint["labels"]
        self.labels: list[str] = [None] * len(label_to_index)
        for sign_id, index in label_to_index.items():
            self.labels[index] = sign_id

        # Which features the model was trained on. Models from before feature versions
        # existed have no key and are version 1. A version this kiosk does not know, or
        # an input size that disagrees with it, is refused rather than fed wrong inputs.
        self.feature_version = int(checkpoint.get("feature_version", 1))
        if self.feature_version not in LIVE_FEATURE_VERSIONS or checkpoint["input_dim"] != INPUT_DIMS[self.feature_version]:
            raise ModelUnavailable(
                f"checkpoint at {self.model_path} uses feature version {self.feature_version} with input size "
                f"{checkpoint['input_dim']}; this kiosk can run versions {LIVE_FEATURE_VERSIONS}"
                + (" (version 3 needs face landmarks, which B2 Perception does not capture yet)"
                   if self.feature_version == 3 else "")
            )

        # Optional per-word commit bars (ml/word_bars.py): words the model is shaky on need more
        # confidence before they are committed without the visitor's confirmation.
        bars = checkpoint.get("word_bars", {})
        if not isinstance(bars, dict) or not all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in bars.values()):
            raise ModelUnavailable(f"checkpoint at {self.model_path} has malformed word_bars")
        self.word_bars: dict[str, float] = dict(bars)

        try:
            self.model = build_model(checkpoint)
        except (KeyError, RuntimeError, ValueError, TypeError) as exc:   # wrong shape, unknown architecture
            raise ModelUnavailable(f"checkpoint at {self.model_path} could not be built: {exc}") from exc

    def commit_bar(self, gloss: str | None) -> float:
        """Confidence a word needs to be committed automatically (below it: offered as a candidate).

        The language-wide bar, raised for words the model is known to confuse. A per-word bar can only
        make the kiosk stricter than the language's calibrated bar, never looser.
        """
        return max(self.confident_threshold, self.word_bars.get(gloss, 0.0)) if gloss else self.confident_threshold

    def warm_up(self) -> float | None:
        """Runs one throwaway forward pass so the first real sign does not pay PyTorch's
        first-call cost (thread pool start-up, buffer allocation), which is the slowest
        moment of a visit. Returns the seconds it took, or None if it could not run: a
        failed warm-up must never stop the kiosk, the real call will report any real fault."""
        import time
        try:
            started = time.perf_counter()
            with torch.no_grad():
                self.model(torch.zeros(1, SEQ_LEN, INPUT_DIMS[self.feature_version]))
            return time.perf_counter() - started
        except Exception:
            return None

    def classify(self, utterance: Utterance) -> Hypothesis:
        clip = self._features(utterance, self.feature_version)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(clip).unsqueeze(0).float())
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        order = np.argsort(probs)[::-1]
        best_label, best_prob = self.labels[order[0]], float(probs[order[0]])
        # The "other" class is never a word to show or a candidate to offer.
        alternatives = [(self.labels[i], float(probs[i])) for i in order
                        if self.labels[i] != OTHER_LABEL][: self.top_k]

        if best_label == OTHER_LABEL or best_prob < self.reject_threshold:
            return Hypothesis(gloss=None, confidence=best_prob, alternatives=alternatives)
        return Hypothesis(gloss=best_label, confidence=best_prob, alternatives=alternatives)

    @staticmethod
    def _signing_seconds(utterance: Utterance) -> float:
        """How long the sign took, measured the way training measured it: first to last frame with a
        visible hand (ml/extract_landmarks.py idle_bounds), not the utterance's own length, which can
        carry idle frames at either end."""
        shown = [lm.ts_ms for lm in utterance.landmarks if lm.left is not None or lm.right is not None]
        times = [lm.ts_ms for lm in utterance.landmarks]
        if not shown or len(times) < 2:
            return max(0.05, (utterance.t_end - utterance.t_start) / 1000.0)
        interval = (times[-1] - times[0]) / (len(times) - 1)            # one frame, in ms
        return max(0.05, (shown[-1] - shown[0] + interval) / 1000.0)

    @staticmethod
    def _features(utterance: Utterance, feature_version: int = 1) -> np.ndarray:
        """Utterance -> the exact (SEQ_LEN, INPUT_DIMS[version]) tensor ml/train.py trained on."""
        frames = [normalise_frame(lm.pose, lm.left, lm.right) for lm in utterance.landmarks]
        clip = resample(frames)          # (SEQ_LEN, FRAME_DIM)
        assert clip.shape[1] == FRAME_DIM
        duration = Recognizer._signing_seconds(utterance) if feature_version in (4, 5) else None
        return model_input(clip, feature_version, duration=duration)
