# features/sequence.py
import numpy as np
from collections import Counter
from tqdm import tqdm
from Bio import SeqIO

from models.base import BaseStep
from data_paths import DataPaths


class SequenceProcessor(BaseStep):
    step_name = "SequenceProcessor"
    produces = ["X_train", "X_test", "train_ids", "test_ids"]

    def __init__(self, dimensions=85):
        self.dimensions = dimensions

    def execute(self):
        train_seqs = self._load_fasta("train_sequences")
        test_seqs = self._load_fasta("testsuperset.fasta")

        train_ids = list(train_seqs.keys())
        test_ids = list(test_seqs.keys())

        X_train = self._extract(list(train_seqs.values()))
        X_test = self._extract(list(test_seqs.values()))

        self.set_ctx("X_train", X_train)
        self.set_ctx("X_test", X_test)
        self.set_ctx("train_ids", train_ids)
        self.set_ctx("test_ids", test_ids)

        return self.context

    # ----------------------------
    def _load_fasta(self, key):
        path = DataPaths.get(key)
        seqs = {}
        for r in SeqIO.parse(path, "fasta"):
            pid = r.id.split("|")[1] if "|" in r.id else r.id
            seqs[pid] = str(r.seq)
        return seqs

    def _extract(self, seqs):
        X = []
        for seq in tqdm(seqs, desc="Extracting features"):
            X.append(self._vectorize(seq))
        return np.asarray(X, dtype=np.float32)

    def _vectorize(self, seq):
        if not seq or len(seq) < 3:
            return np.zeros(self.dimensions)

        c = Counter(seq)
        n = len(seq)
        freq = [c.get(a, 0) / n for a in "ACDEFGHIKLMNPQRSTVWY"]
        vec = np.array(freq, dtype=np.float32)
        return np.pad(vec, (0, self.dimensions - len(vec)))
