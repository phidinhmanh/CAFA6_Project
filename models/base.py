from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseStep(ABC):
    """
    Base class cho mọi pipeline step
    """

    step_name: str | None = None

    # context keys mà step tạo ra
    produces: list[str] = []

    # nếu step tạo file output → khai báo ở đây
    output_file_key: str | None = None  # vd: "prediction_path"

    def run(self, context: dict, config: dict) -> dict:
        self.context = context or {}
        self.config = config or {}

        if self._already_done():
            print(f"--- SKIP STEP: {self.step_name or self.__class__.__name__}")
            return self.context

        print(f"\n=== RUNNING STEP: {self.step_name or self.__class__.__name__} ===")
        out = self.execute() or {}
        self.context.update(out)
        return self.context

    # ===============================
    # SKIP LOGIC (FIX Ở ĐÂY)
    # ===============================
    def _already_done(self) -> bool:
        """
        Step được coi là DONE nếu:
        - output_file_key tồn tại trong context
        - file đó tồn tại trên disk
        """
        if not self.output_file_key:
            return False

        path = self.context.get(self.output_file_key)
        if not path:
            return False

        p = Path(path)
        if not p.exists():
            return False

        # CRITICAL RULE:
        # submission.tsv KHÔNG được dùng để skip step trung gian
        if p.name == "submission.tsv":
            return False

        return True

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

    # ---------- prediction helpers ----------
    def get_input_path(self):
        return self.get_ctx("prediction_path")

    def set_output_path(self, path):
        self.set_ctx("prediction_path", path)
