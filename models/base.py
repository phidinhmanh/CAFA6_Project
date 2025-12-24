from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseStep(ABC):
    """
    Base class cho mọi pipeline step
    """

    step_name: str | None = None

    def run(self, context: dict, config: dict) -> dict:
        self.context = context or {}
        self.config = config or {}

        print(f"\n=== RUNNING STEP: {self.step_name or self.__class__.__name__} ===")
        out = self.execute() or {}
        self.context.update(out)
        return self.context

    # ===============================
    # ABSTRACT
    # ===============================
    @abstractmethod
    def execute(self) -> dict:
        pass

    # ===============================
    # COMMON HELPERS
    # ===============================
    def get_cfg(self, key, default=None):
        return self.config.get(key, default)

    def get_ctx(self, key, default=None):
        return self.context.get(key, default)

    def set_ctx(self, key, value):
        self.context[key] = value

    # ---------- IO helpers ----------
    def read_tsv(self, path, names=("id", "term", "score")):
        return pd.read_csv(path, sep="\t", header=None, names=list(names))

    def write_tsv(self, rows, path):
        with open(path, "w") as f:
            f.writelines(rows)
        return path

    def set_output_path(self, path):
        self.set_ctx("prediction_path", path)
