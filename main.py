# main.py
import yaml
from factory import Factory
from runner import PipelineRunner
from data_paths import DataPaths


def main():
    print("--- STARTING CAFA PIPELINE ---")

    # 1. Resolve data paths
    DataPaths.autopath("/kaggle/input")

    # 2. Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # 3. Load existing context if any
    context = {}

    # Nếu đã có submission → inject luôn
    if DataPaths.exists("submission"):
        context["prediction_path"] = DataPaths.get("submission")
        print(">>> Found existing submission, will skip model steps")

    runner = PipelineRunner(config, context)

    # 4. Build steps dynamically
    steps = []
    for _, step_cfg in config["pipeline"].items():
        steps.append(Factory.create(step_cfg))

    # 5. Run
    runner.run(steps)

    print("\nSUCCESS! Pipeline finished.")


if __name__ == "__main__":
    main()
