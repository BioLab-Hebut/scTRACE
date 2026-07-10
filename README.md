# scTRACE

scTRACE is a single-cell rare cell identification framework that combines deep representation learning, cluster refinement, and anomaly-based rare cluster detection.

## Files

- `demo.py`: end-to-end example for training and rare cell detection.
- `trace_model.py`: neural network modules, including the convolutional autoencoder and attention blocks.
- `trace_core.py`: clustering training, cluster refinement, and rare cell detection logic.
- `metrics.py`: clustering metric utilities.
- `preprocess.py`: preprocessing for CSV, TXT, and 10x MTX inputs.


## Requirements

Install the packages listed in `requirements.txt`. CUDA is required for the default demo configuration.

## Data Format

The demo expects the following files under the configured data directory:

- `processed_data.csv`: normalized expression matrix with 784 selected features.
- `processed_data_raw_counts.csv`: raw count matrix aligned with `processed_data.csv`.
- `celltypes.csv`: label table containing a numeric `true_label` column. An optional `cell_type` column is used for readable label summaries.

## Run

Edit `DATA_PATH` and `RESULTS_PATH` in `demo.py`, then run:

```bash
python demo.py
```

The demo prints training progress, anomaly detection progress, the number of rare clusters found, and the label composition of each rare cluster.

## Outputs

Results are written to the configured results directory with the `rare_cell_trace_*` filename prefix.
