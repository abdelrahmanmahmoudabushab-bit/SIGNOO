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
from core.features import FRAME_DIM, add_velocity, normalise_frame, resample

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
        from model import SignTransformer  # ml/model.py, added to sys.path above

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

        required = {"model", "labels", "input_dim"}
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

        self.model = SignTransformer(input_dim=checkpoint["input_dim"], num_classes=len(self.labels))
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def classify(self, utterance: Utterance) -> Hypothesis:
        clip = self._features(utterance)
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
    def _features(utterance: Utterance) -> np.ndarray:
        """Utterance -> the exact (SEQ_LEN, FRAME_DIM*2) tensor ml/train.py trained on."""
        frames = [normalise_frame(lm.pose, lm.left, lm.right) for lm in utterance.landmarks]
        clip = resample(frames)          # (SEQ_LEN, FRAME_DIM)
        assert clip.shape[1] == FRAME_DIM
        return add_velocity(clip)        # (SEQ_LEN, FRAME_DIM*2)
