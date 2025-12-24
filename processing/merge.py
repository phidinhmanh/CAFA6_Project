import pandas as pd
from models.base import BaseStep


class Merge(BaseStep):
    step_name = "Merge"
    output_file_key = "prediction_path"

    def __init__(
        self, method="max", merge_path="submission.tsv", extra_predictions=None
    ):
        self.method = method
        self.merge_path = merge_path
        self.extra_predictions = extra_predictions or []

    def execute(self):
        # 1. Base prediction (from previous step)
        base_path = self.get_ctx("prediction_path")
        if not base_path:
            raise RuntimeError("Merge requires 'prediction_path' in context")

        dfs = [self.read_tsv(base_path)]

        # 2. Extra predictions (BLAST, ensemble, ...)
        for p in self.extra_predictions:
            if p == self.merge_path:
                # tránh tự merge chính output
                continue
            dfs.append(self.read_tsv(p))

        # 3. Merge
        merged = pd.concat(dfs, ignore_index=True)

        if self.method == "max":
            final = merged.groupby(["id", "term"], as_index=False)["score"].max()
        elif self.method == "mean":
            final = merged.groupby(["id", "term"], as_index=False)["score"].mean()
        else:
            raise ValueError(f"Unknown merge method: {self.method}")

        # 4. Write output
        final.to_csv(self.merge_path, sep="\t", header=False, index=False)

        # 5. Update pipeline context
        self.set_output_path(self.merge_path)

        return {"prediction_path": self.merge_path}
