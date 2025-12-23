# Protein Predict

## Overview
This project provides a Python framework for protein prediction pipelines. It features an automated data path discovery system and a modular step-based architecture for processing tasks.

## Components

### DataPaths (`data_paths.py`)
A utility class designed to simplify file management, particularly useful in environments like Kaggle or local development where paths may vary.

- **Auto-discovery**: Recursively scans directories (default: `/kaggle/input` or user home) to register files.
- **Smart Resolution**: Retrieves file paths using exact names or unique substrings, eliminating the need for hardcoded absolute paths.

### BaseStep (`models/base.py`)
An abstract base class for defining pipeline steps.

- **Context Awareness**: Steps operate on a shared `context` dictionary.
- **Automatic Skipping**: Skips execution if the expected output keys (`produces`) are already present in the context.
- **I/O Helpers**: Includes utility methods for reading and writing TSV files and managing input/output paths.

## Usage

### Managing Data Paths
```python
from data_paths import DataPaths

# Initialize and scan for specific file types
DataPaths.autopath(suffixes=[".fasta", ".tsv", ".csv"])

# Get a file path (throws error if ambiguous or not found)
input_file = DataPaths.get("dataset_v1.csv")
```

### Creating a Pipeline Step
```python
from models.base import BaseStep

class PredictionStep(BaseStep):
    step_name = "ModelPrediction"
    produces = ["prediction_scores"]

    def execute(self) -> dict:
        # Logic to run prediction
        input_path = self.get_input_path()
        
        # Update context with results
        self.set_ctx("prediction_scores", [0.95, 0.88])
        return self.context
```