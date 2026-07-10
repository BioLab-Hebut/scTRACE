import os

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse


MATRIX_FILE_PATH = "/home/jinrongqi/scTRACE-1/data/Hrvatin/data.h5"
LABEL_FILE_PATH = None
LABEL_COLUMN_NAME = None

OUTPUT_DATA_PATH = "/home/jinrongqi/scTRACE-1/data/Hrvatin/processed_data.csv"
OUTPUT_LABEL_PATH = "/home/jinrongqi/scTRACE-1/data/Hrvatin/celltypes.csv"
USE_READ_COUNTING = False

N_TOP_GENES = 784
MIN_CELLS = 3


def normalize_labels(labels):
    if labels is None or len(labels) == 0:
        return None

    cell_types = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in labels
    ]
    unique_labels = sorted(set(cell_types))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_id[label] for label in cell_types]

    return pd.DataFrame({
        "true_label": numeric_labels,
        "cell_type": cell_types,
    })


def load_external_labels(label_file_path, label_column_name):
    if label_file_path is None:
        return None
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"Label file not found: {label_file_path}")

    labels_df = pd.read_csv(label_file_path)
    if label_column_name is not None:
        if label_column_name not in labels_df.columns:
            raise ValueError(f"Label column not found: {label_column_name}")
        return labels_df[label_column_name].values

    candidate_columns = [
        "label",
        "cell_type",
        "celltype",
        "type",
        "cluster",
        "class",
        "annotation",
    ]
    for column in candidate_columns:
        if column in labels_df.columns:
            return labels_df[column].values

    return labels_df.iloc[:, 0].values


def load_h5_matrix(file_path):
    with h5py.File(file_path, "r") as h5_file:
        if "exprs" in h5_file:
            required_fields = ["exprs/data", "exprs/indices", "exprs/indptr", "exprs/shape"]
            missing_fields = [field for field in required_fields if field not in h5_file]
            if missing_fields:
                raise ValueError(f"Incomplete sparse H5 matrix: {missing_fields}")

            values = np.asarray(h5_file["exprs/data"][:])
            indices = np.asarray(h5_file["exprs/indices"][:])
            indptr = np.asarray(h5_file["exprs/indptr"][:])
            shape = tuple(h5_file["exprs/shape"][:])
            matrix = csr_matrix((values, indices, indptr), shape=shape)
        elif "X" in h5_file:
            matrix = np.asarray(h5_file["X"][:])
        elif "data" in h5_file:
            matrix = np.asarray(h5_file["data"][:])
        else:
            raise ValueError("H5 file must contain 'exprs', 'X', or 'data'")

        labels = np.asarray(h5_file["Y"][:]) if "Y" in h5_file else None

    return matrix, labels


def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Matrix file not found: {file_path}")

    extension = os.path.splitext(file_path)[1].lower()
    labels = None

    if extension == ".csv":
        matrix = pd.read_csv(file_path, header=0, index_col=0).T
    elif extension == ".txt":
        matrix = pd.read_csv(file_path, header=0, index_col=0, delim_whitespace=True).T
    elif extension in {".h5", ".hdf5"}:
        matrix, labels = load_h5_matrix(file_path)
    else:
        raise ValueError(f"Unsupported matrix format: {extension}")

    return sc.AnnData(matrix, dtype=np.float32), labels


def clean_invalid_values(data):
    if issparse(data.X):
        data.X.data = np.nan_to_num(data.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        data.X = np.nan_to_num(data.X, nan=0.0, posinf=0.0, neginf=0.0)


def save_outputs(data, raw_counts_df, labels_df, output_data_path, output_label_path):
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

    data.to_df().to_csv(output_data_path)
    raw_counts_path = output_data_path.replace(".csv", "_raw_counts.csv")
    raw_counts_df.to_csv(raw_counts_path)

    if labels_df is not None:
        labels_df.to_csv(output_label_path, index=False)

    return raw_counts_path


def preprocess():
    data, h5_labels = load_matrix(MATRIX_FILE_PATH)

    raw_labels = (
        load_external_labels(LABEL_FILE_PATH, LABEL_COLUMN_NAME)
        if LABEL_FILE_PATH is not None
        else h5_labels
    )
    labels_df = normalize_labels(raw_labels)
    if labels_df is not None and len(labels_df) != data.shape[0]:
        raise ValueError(
            f"Label count ({len(labels_df)}) does not match cell count ({data.shape[0]})"
        )

    sc.pp.filter_genes(data, min_cells=MIN_CELLS)
    raw_counts_df = np.maximum(np.round(data.to_df()), 0).astype(int)

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    clean_invalid_values(data)

    if USE_READ_COUNTING:
        rounded_df = np.round(data.to_df())
        data = sc.AnnData(rounded_df, dtype=np.float32, obs=data.obs)
        sc.pp.highly_variable_genes(
            data,
            n_top_genes=N_TOP_GENES,
            flavor="seurat_v3",
            subset=True,
        )
    else:
        sc.pp.highly_variable_genes(data, n_top_genes=N_TOP_GENES, subset=True)

    data = data[:, data.var.highly_variable].copy()
    raw_counts_df = raw_counts_df[data.var_names.tolist()]

    raw_counts_path = save_outputs(
        data,
        raw_counts_df,
        labels_df,
        OUTPUT_DATA_PATH,
        OUTPUT_LABEL_PATH,
    )

    print(f"Normalized data: {OUTPUT_DATA_PATH}")
    print(f"Raw counts: {raw_counts_path}")
    if labels_df is not None:
        print(f"Cell labels: {OUTPUT_LABEL_PATH}")


if __name__ == "__main__":
    preprocess()
