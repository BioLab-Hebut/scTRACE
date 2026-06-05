# scTRACE
Single-cell Transcriptomic Rare-cell identification with Anomaly-based Cluster rEfinement

## Overview
scTRACE is a Python framework for rare cell detection from single-cell RNA-seq data, based on anomaly detection and cluster refinement strategy, a three-stage framework comprising Deep Clustering for Initial Clustering Assignment, Decomposition and Merging for Cluster Refinement, and Anomaly Scoring for Rare Cell Identification.

## Environment Requirements
```
numpy                     1.22.4   
python                    3.9.23   
pandas                    1.4.2  
scanpy                    1.9.1  
scipy                     1.13.1 
seaborn                   0.13.2 
torch                     2.8.0+cu129
```
## Project Structure
```
├── preprocess.py   # Single-cell data preprocessing
├── network.py      # Deep network architecture of scTRACE
├── Metrics.py      # Evaluation metrics of clustering
├── sc.py           # Main entry of scTRACE model
└── demo.py         # Reproducible demo script
```
