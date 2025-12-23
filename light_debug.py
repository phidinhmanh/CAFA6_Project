# light_debug.py
import yaml
import os
from factory import Factory
from runner import PipelineRunner
from data_paths import DataPaths


def main():
    print("--- STARTING LIGHT DEBUG SCRIPT ---")

    # 1. Resolve data paths to our mock data directory
    # Using absolute path to ensure certainty
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mock_data_path = os.path.join(base_dir, "tmp_data")

    print(f"Scanning mock data in: {mock_data_path}")
    DataPaths.autopath(mock_data_path, refresh=True)

    # 2. Load config
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 3. Initialize context
    context = {}

    # 4. Build and run steps dynamically using the runner
    runner = PipelineRunner(config, context)

    steps = []
    print("\n--- Building Pipeline Steps ---")
    for name, step_cfg in config["pipeline"].items():
        print(f"  + Adding step: {name} ({step_cfg.get('class')})")
        steps.append(Factory.create(step_cfg))

    print("\n--- Running Pipeline ---")
    runner.run(steps)

    # 5. Verify outputs
    print("\n--- Verifying Outputs ---")
    if "prediction_path" in context:
        out_path = context["prediction_path"]
        print(f"Final prediction path: {out_path}")
        if os.path.exists(out_path):
            with open(out_path) as f:
                lines = f.readlines()
                print(f"Output contains {len(lines)} lines.")
                print("First 5 lines:")
                for line in lines[:5]:
                    print(f"  {line.strip()}")
        else:
            print(f"WARNING: Output file {out_path} not found!")

    print("\nSUCCESS! Light debug script finished.")


if __name__ == "__main__":
    main()
