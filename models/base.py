# models/base.py
from abc import ABC, abstractmethod
import pandas as pd


class BaseStep(ABC):
    """
    Base class cho mọi pipeline step
    """

    step_name: str = None  # type: ignore

    produces = []  # context keys mà step tạo ra

    def run(self, context: dict, config: dict) -> dict:
        self.context = context or {}
        self.config = config or {}

        if self._already_done():
            print(f"--- SKIP STEP: {self.step_name or self.__class__.__name__}")
            return self.context

        print(f"\n=== RUNNING STEP: {self.step_name or self.__class__.__name__} ===")
        return self.execute() or self.context

    def _already_done(self) -> bool:
        return self.produces and all(k in self.context for k in self.produces)

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

    def get_input_path(self):
        """
        Chuẩn hoá input prediction path
        """
        return self.get_ctx("prediction_path")

    def set_output_path(self, path):
        self.set_ctx("prediction_path", path)
