# processing/merge.py
import pandas as pd
from models.base import BaseStep


class Merge(BaseStep):
    step_name = "Merge"

    def __init__(self, method="max"):
        self.method = method

    def execute(self):
        base_path = self.get_ctx("prediction_path")
        base_df = self.read_tsv(base_path)

        dfs = [base_df]

        extra = self.get_cfg("extra_predictions", [])
        for p in extra: # type: ignore
            dfs.append(self.read_tsv(p))

        merged = pd.concat(dfs, ignore_index=True)

        if self.method == "max":
            final = merged.groupby(["id", "term"], as_index=False)["score"].max()
        elif self.method == "mean":
            final = merged.groupby(["id", "term"], as_index=False)["score"].mean()
        else:
            raise ValueError("Unknown merge method")

        out = self.get_cfg("final_output", "submission.tsv")
        final.to_csv(out, sep="\t", header=False, index=False)

        self.set_ctx("prediction_path", out)
        return self.context
