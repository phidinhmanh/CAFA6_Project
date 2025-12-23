# Protein Function Prediction Pipeline

A modular pipeline for [CAFA 6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) protein function prediction.

## Quick Start

```bash
# Install dependencies
uv sync

# Run with mock data (5% sampled)
uv run python light_debug.py

# Run full pipeline (Kaggle or local)
uv run python main.py
```

## Project Structure

```
├── main.py                 # Full pipeline entry
├── light_debug.py          # Debug with sampled data
├── config.yaml             # Pipeline configuration
├── data_paths.py           # Auto file discovery
├── factory.py              # Dynamic step creation
├── runner.py               # Pipeline executor
├── features/
│   └── sequence.py         # Sequence feature extraction
├── models/
│   ├── base.py             # Abstract base step
│   └── generic.py          # Sklearn wrapper
├── processing/
│   ├── ontology_filter.py  # GO term filtering
│   └── merge.py            # Result merging
└── tmp_data/               # Sampled test data (5%)
```

## Pipeline Steps

| Step | Class | Description |
|------|-------|-------------|
| 1 | `SequenceProcessor` | Extract sequence features |
| 2 | `SklearnWrapper` | KNN prediction |
| 3 | `OntologyFilter` | Taxonomy + GO propagation |
| 4 | `Merge` | Combine results |

## Configuration

Edit `config.yaml` to modify:
- Model parameters (`n_neighbors`, `metric`)
- Filter thresholds (`min_score`, `top_k`)
- File paths

## Data Files

The pipeline expects these files (auto-discovered via `DataPaths`):

| File | Description |
|------|-------------|
| `train_sequences.fasta` | Training sequences |
| `train_terms.tsv` | GO term annotations |
| `train_taxonomy.tsv` | Taxon mappings |
| `go-basic.obo` | Gene Ontology |
| `testsuperset.fasta` | Test sequences |
| `testsuperset-taxon-list.tsv` | Test taxon mappings |

## Local Testing

Generate 5% sampled data for quick testing:

```bash
# Data is auto-sampled from Downloads/cafa-6-protein-function-prediction
uv run python sample_real_data.py
```