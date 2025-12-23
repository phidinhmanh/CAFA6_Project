import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from data_paths import DataPaths
from models.base import BaseStep


class KNNModel(BaseStep):
    def __init__(self, n_neighbors=7, metric="cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def run(self, context, config):
        print(f"[{self.__class__.__name__}] Processing Labels...")

        # 1. Load Labels dynamically
        terms_path = DataPaths.get("train_terms")
        train_terms = pd.read_csv(terms_path, sep="\t")

        # 2. Align Labels with Train IDs from Context
        train_ids = context["train_ids"]
        # Logic to build y_train (Top GO terms)
        top_n = config.get("top_go_terms", 2000)
        top_go = train_terms["term"].value_counts().head(top_n).index
        go2idx = {go: i for i, go in enumerate(top_go)}

        y_train = np.zeros((len(train_ids), len(top_go)), dtype=np.float32)
        # Fast lookup
        labels_map = train_terms.groupby("EntryID")["term"].apply(set).to_dict()

        for i, pid in enumerate(train_ids):
            if pid in labels_map:
                for go in labels_map[pid]:
                    if go in go2idx:
                        y_train[i, go2idx[go]] = 1

        # 3. Train
        print(f"[{self.__class__.__name__}] Fitting KNN (k={self.n_neighbors})...")
        nn = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=-1
        )
        nn.fit(context["X_train"])

        # 4. Predict
        # Instead of writing to file immediately, store in memory or temp file
        # To avoid OOM, let's write to a generic 'intermediate' file defined in config or default
        output_file = config.get("knn_output", "knn_predictions.tsv")
        print(f"[{self.__class__.__name__}] Predicting -> {output_file}...")

        X_test = context["X_test"]
        test_ids = context["test_ids"]
        batch_size = config.get("batch_size", 1024)

        with open(output_file, "w") as f:
            for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
                batch_X = X_test[i : i + batch_size]
                dists, idxs = nn.kneighbors(batch_X)
                weights = 1 / (dists + 1e-8)
                weights /= weights.sum(axis=1, keepdims=True)  # Normalize

                # Weighted vote
                preds = (y_train[idxs] * weights[:, :, None]).sum(axis=1)

                # Write top results
                for j, pid in enumerate(test_ids[i : i + batch_size]):
                    scores = preds[j]
                    # Get top indices
                    top_indices = scores.argsort()[::-1][
                        :100
                    ]  # Keep top 100 for efficiency
                    for k in top_indices:
                        if scores[k] > 0.01:
                            f.write(f"{pid}\t{top_go[k]}\t{scores[k]:.3f}\n")

        context["knn_path"] = output_file
        return context
