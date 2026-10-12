"""Text embeddings for every sign's name, in English and Arabic, from an open multilingual model.

    python ml/label_embeddings.py --out data/asl_all/label_text.npz

`ml/train.py --text-embeddings` turns these into the classifier's class prototypes (SignNet's
`class_text` buffer), so a word with a handful of clips is placed near words that mean something
similar, and `dog` and `puppy` are not unrelated output units. Only training needs this model: the
finished checkpoint carries the vectors, so the kiosk loads no NLP library.

Model: sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0), 50+ languages
including Arabic. A word's vector is the normalised mean of its English and Arabic name vectors,
so the two languages' names pull toward one shared meaning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def names(vocab_path: Path, words_path: Path) -> dict[str, tuple[str, str]]:
    """sign id (lower-case) -> (english, arabic)."""
    out = {}
    for entry in json.loads(vocab_path.read_text(encoding="utf-8")):
        out[entry["id"].lower()] = (entry["en"], entry["ar"])
    for gid, ar in json.loads(words_path.read_text(encoding="utf-8")).items():
        out.setdefault(gid.lower(), (gid, ar))
    return out


def embed(texts: list[str], model_name: str = MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=str(ROOT / "core/vocab_asl.json"))
    ap.add_argument("--words", default=str(ROOT / "ml/asl_citizen_words.json"))
    args = ap.parse_args(argv)

    table = names(Path(args.vocab), Path(args.words))
    ids = sorted(table)
    vectors = embed([table[i][0] for i in ids] + [table[i][1] for i in ids])
    en, ar = vectors[: len(ids)], vectors[len(ids):]
    mean = en + ar
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-9)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(ids), vectors=mean.astype(np.float32))
    print(f"{len(ids)} words -> {args.out}  ({mean.shape[1]}-d)")
    # Sanity: English and Arabic names of the same word should agree far more than chance.
    agreement = float(np.mean(np.sum(en * ar, axis=1)))
    chance = float(np.mean(en @ ar.T))
    print(f"mean en/ar cosine for the same word {agreement:.2f} (any pair {chance:.2f})")


if __name__ == "__main__":
    main()
