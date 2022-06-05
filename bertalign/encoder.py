import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps


# See other cross-lingual embedding models at
# https://www.sbert.net/docs/pretrained_models.html
class Encoder:
    def __init__(self, model_name="sentence-transformers/LaBSE", cpu_only=False):
        print(f"Loading model {model_name} ...")
        self.model = SentenceTransformer(
            model_name,
            device="cpu" if cpu_only or not torch.cuda.is_available() else "cuda",
        )
        self.model_name = model_name

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(overlaps)
        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs
