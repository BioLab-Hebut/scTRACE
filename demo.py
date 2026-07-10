import os
import random
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "6"

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import Dataset

from trace_model import TraceAutoEncoder, TraceBottleneck
from trace_core import TraceClusterTrainer, TraceDetector


DATA_PATH = "/home/jinrongqi/scTRACE/test/"
RESULTS_PATH = "/home/jinrongqi/scTRACE/results/"


class TraceDataset(Dataset):
    def __init__(self, norm_data, raw_counts_data, labels, transform=None):
        self.transform = transform
        self.norm_data = self._load_data(norm_data, labels)
        self.raw_counts_data = self._load_data(raw_counts_data, labels)

    def _load_data(self, data, labels):
        values = np.asarray(data, dtype="float32").reshape((-1, 28, 28))
        label_values = labels["true_label"].to_numpy(dtype=int)
        return list(zip(values, label_values.tolist()))

    def __len__(self):
        return len(self.norm_data)

    def __getitem__(self, index):
        norm_image, label = self.norm_data[index]
        raw_image, _ = self.raw_counts_data[index]
        if self.transform:
            norm_image = self.transform(norm_image)
            raw_image = self.transform(raw_image)
        return norm_image, raw_image, label, index


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def weights_init(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


def load_inputs(data_path):
    required_files = ["processed_data.csv", "processed_data_raw_counts.csv", "celltypes.csv"]
    for filename in required_files:
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    labels = pd.read_csv(os.path.join(data_path, "celltypes.csv"), header=0, index_col=None)
    if "true_label" not in labels.columns:
        raise ValueError("celltypes.csv must contain a numeric 'true_label' column")

    norm_data = pd.read_csv(os.path.join(data_path, "processed_data.csv"), header=0, index_col=0)
    raw_counts = pd.read_csv(os.path.join(data_path, "processed_data_raw_counts.csv"), header=0, index_col=0)

    if len(labels) != norm_data.shape[0]:
        raise ValueError(f"Label count ({len(labels)}) does not match cell count ({norm_data.shape[0]})")
    if norm_data.shape != raw_counts.shape:
        raise ValueError(f"Data shape mismatch: normalized {norm_data.shape} vs raw counts {raw_counts.shape}")
    if not norm_data.index.equals(raw_counts.index):
        raise ValueError("Cell names mismatch between normalized and raw counts data")
    if not norm_data.columns.equals(raw_counts.columns):
        raise ValueError("Gene names mismatch between normalized and raw counts data")
    if norm_data.shape[1] != 784:
        raise ValueError(f"Expected 784 features, got {norm_data.shape[1]}. Please run preprocessing first.")

    num_cluster = labels["true_label"].nunique()
    return norm_data, raw_counts, labels, num_cluster


def build_data_loader(norm_data, raw_counts, labels, batch_size):
    dataset = TraceDataset(
        norm_data,
        raw_counts,
        labels,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )


def build_model(data_loader, dataset_size, num_cluster, config):
    ae = TraceAutoEncoder(TraceBottleneck, [1, 1, 1]).cuda()
    ae.apply(weights_init)
    return TraceClusterTrainer(
        ae,
        data_loader,
        dataset_size,
        batch_size=config["batch_size"],
        pretraining_epoch=config["pretraining_epoch"],
        MaxIter1=config["max_iter"],
        num_cluster=num_cluster,
        m=config["m"],
        T1=config["update_interval"],
        latent_size=config["latent_size"],
        dataset_name=config["dataset_name"],
        a=config["alpha"],
    )


def build_label_lookup(cell_names, labels):
    label_column = "cell_type" if "cell_type" in labels.columns else "true_label"
    label_values = labels[label_column].astype(str).tolist()
    return dict(zip(cell_names, label_values))


def print_rare_cluster_summary(result, label_lookup):
    print(f"Rare clusters found: {len(result)}")
    for idx, cluster in enumerate(result, start=1):
        cluster_labels = [label_lookup[cell] for cell in cluster if cell in label_lookup]
        label_counts = Counter(cluster_labels)
        composition = ", ".join(f"{label}: {count}" for label, count in label_counts.most_common())
        if not composition:
            composition = "label information unavailable"
        print(f"Rare cluster {idx}: {len(cluster)} cells")
        print(f"Label composition: {composition}")




@contextmanager
def suppress_output(enabled):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def main(data_path=DATA_PATH, results_path=RESULTS_PATH, quiet=False):
    set_seed(100)
    os.makedirs(results_path, exist_ok=True)

    config = {
        "batch_size": 100,
        "pretraining_epoch": 20,
        "max_iter": 20,
        "update_interval": 2,
        "m": 1.5,
        "latent_size": 10,
        "dataset_name": "rare_cell",
        "alpha": 0.2,
        "merge_h": 50,
        "overlap_h": 0.7,
        "rare_h": 0.01,
        "seed": 2023,
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo")

    norm_data, raw_counts, labels, num_cluster = load_inputs(data_path)
    data_loader = build_data_loader(norm_data, raw_counts, labels, config["batch_size"])
    deep_cluster = build_model(data_loader, norm_data.shape[0], num_cluster, config)

    with suppress_output(quiet):
        ari_list, nmi_list, acc_list = [], [], []
        if config["pretraining_epoch"]:
            deep_cluster.pretrain()
        if config["max_iter"]:
            ari_list, nmi_list, acc_list = deep_cluster.train_deep_clustering()

        deep_cluster_labels = deep_cluster.predict_clusters()
        print("Anomaly detection started.")
        result, score, sub_clusters, degs_list = TraceDetector().detect_rare_cells(
            X_norm=norm_data.values.astype("float32"),
            cellNames=norm_data.index.tolist(),
            geneNames=norm_data.columns.tolist(),
            deep_init_clusters=deep_cluster_labels,
            dataName=config["dataset_name"],
            save_path=results_path,
            merge_h=config["merge_h"],
            overlap_h=config["overlap_h"],
            rare_h=config["rare_h"],
            seed=config["seed"],
            save_full=True,
        )

    print_rare_cluster_summary(result, build_label_lookup(norm_data.index.tolist(), labels))

    return {
        "clustering_metrics": {
            "ARI": ari_list,
            "NMI": nmi_list,
            "ACC": acc_list,
        },
        "rare_cell_results": {
            "rare_clusters": result,
            "anomaly_scores": score,
            "sub_clusters": sub_clusters,
            "degs_list": degs_list,
        },
        "deep_cluster_labels": deep_cluster_labels,
        "summary": {
            "total_cells": len(norm_data.index),
            "total_genes": len(norm_data.columns),
            "num_clusters_detected": num_cluster,
            "deep_clustering_clusters": len(np.unique(deep_cluster_labels)),
            "final_subclusters": len(np.unique(sub_clusters)) if len(sub_clusters) > 0 else 0,
            "rare_clusters_found": len(result),
            "total_rare_cells": sum(len(cluster) for cluster in result) if result else 0,
            "best_ARI": max(ari_list) if ari_list else 0,
            "best_NMI": max(nmi_list) if nmi_list else 0,
            "best_ACC": max(acc_list) if acc_list else 0,
            "used_raw_counts": True,
        },
    }


if __name__ == "__main__":
    main()
