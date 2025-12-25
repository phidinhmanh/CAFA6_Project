# models/generic.py
import importlib
import pandas as pd
from tqdm import tqdm

from models.base import BaseStep
from data_paths import DataPaths


class SklearnWrapper(BaseStep):
    step_name = "SklearnModel"
    produces = {}

    def __init__(self, library, algorithm, model_params=None):
        self.library = library
        self.algorithm = algorithm
        self.model_params = model_params or {}

        module = importlib.import_module(library)
        self.model_cls = getattr(module, algorithm)
        self.model = self.model_cls(**self.model_params)

    def _already_done(self):
        if self.get_ctx("prediction_path"):
            return True
        return False

    def execute(self):
        if self._already_done():
            print(
                f"--- SKIP STEP (cached): {self.step_name or self.__class__.__name__}"
            )
            return self.context

        X_train = self.get_ctx("X_train")

        print(f">> Fitting {self.algorithm}")
        self.model.fit(X_train)

        if hasattr(self.model, "kneighbors"):
            return self._predict_neighbors()
        elif hasattr(self.model, "predict_proba"):
            return self._predict_proba()
        else:
            raise RuntimeError("Unsupported sklearn API")

    # ----------------------------
    def _predict_neighbors(self):
        train_terms = pd.read_csv(DataPaths.get("train_terms"), sep="\t")
        label_map = train_terms.groupby("EntryID")["term"].apply(list).to_dict()

        train_ids = self.get_ctx("train_ids")
        test_ids = self.get_ctx("test_ids")
        X_test = self.get_ctx("X_test")

        output = []

        for i in tqdm(range(len(X_test)), desc="Predict"):  # type: ignore
            dists, idxs = self.model.kneighbors([X_test[i]])  # type: ignore
            weights = 1 / (dists[0] + 1e-8)

            scores = {}
            for w, idx in zip(weights, idxs[0]):
                pid = train_ids[idx]  # type: ignore
                for t in label_map.get(pid, []):
                    scores[t] = scores.get(t, 0) + w

            norm = sum(weights)
            for t, s in scores.items():
                prob = (s / norm) **1.5
                if prob > 0.01:
                    output.append(f"{test_ids[i]}\t{t}\t{prob:.3f}\n")  # type: ignore

        path = "model_output.tsv"
        self.write_tsv(output, path)
        self.set_ctx("prediction_path", path)
        return self.context

    def _predict_proba(self):
        # generic classifier support (future-proof)
        probs = self.model.predict_proba(self.get_ctx("X_test"))
        test_ids = self.get_ctx("test_ids")
        output = []
        for i, pid in enumerate(test_ids):  # type: ignore
            for j, prob in enumerate(probs[i]):
                output.append(f"{pid}\t{self.model.classes_[j]}\t{prob:.3f}\n")
        path = "model_output.tsv"
        self.write_tsv(output, path)
        self.set_ctx("prediction_path", path)
        return self.context
