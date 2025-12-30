# 🧬 Protein Function Prediction Pipeline (CAFA 6)
This repository features a robust, automated machine learning pipeline designed for the CAFA 6 competition. The system predicts biological functions of proteins (Gene Ontology terms) using advanced sequence-based feature extraction and hierarchical post-processing.

# 🌟 Technical Highlights
Modular Architecture: Built with a decoupled design, allowing independent development and testing of feature extractors, models, and post-processors.

Biological Logic Integration: Unlike standard ML pipelines, this system includes an Ontology Filter that ensures predictions respect the hierarchical structure (parent-child relationships) of Gene Ontology.

Modern Engineering Stack: Powered by uv, the next-generation Python package manager, ensuring reproducible environments and ultra-fast dependency resolution.

Scalable Development Workflow: Includes an automated Sampling Engine (light_debug.py) that creates a 5% representative subset of massive biological datasets for rapid prototyping and local testing.

# 🛠 The Pipeline Workflow
The project follows a structured 4-stage execution flow:

Feature Engineering: Extracts numerical representations from raw protein FASTA sequences.

Predictive Modeling: Implements an optimized K-Nearest Neighbors (KNN) approach via a custom SklearnWrapper for high-dimensional data.

Domain-Specific Processing: Applies Taxonomy mapping and GO Propagation to refine scores based on biological constraints.

Result Aggregation: Merges multi-source predictions into a standardized format ready for large-scale evaluation.

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
