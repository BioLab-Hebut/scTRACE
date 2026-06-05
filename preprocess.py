import numpy as np
import pandas as pd
import scanpy as sc
import os
import argparse
import h5py
from scipy.sparse import issparse, csr_matrix

MATRIX_FILE_PATH = '/home/jinrongqi/scSMD-1/data/Hrvatin/data.h5'  

LABEL_FILE_PATH = None   
LABEL_COLUMN_NAME = None   

OUTPUT_DATA_PATH = '/home/jinrongqi/scSMD-1/data/Hrvatin/processed_data.csv'  
OUTPUT_LABEL_PATH = '/home/jinrongqi/scSMD-1/data/Hrvatin/celltypes.csv'      

USE_READ_COUNTING = False


def process_labels_unified(original_labels, label_source="unknown"):
    if len(original_labels) == 0:
        return [], [], {}
    
    if isinstance(original_labels[0], bytes):
        cell_types = [label.decode('utf-8') for label in original_labels]
    elif isinstance(original_labels[0], (int, float)):
        cell_types = [str(label) for label in original_labels]
    else:
        cell_types = [str(label) for label in original_labels]
    
    unique_labels = sorted(list(set(cell_types)))
    
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    true_labels = [label_mapping[label] for label in cell_types]
    
    for original, new_idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        count = cell_types.count(original)
        percentage = count / len(cell_types) * 100

    
    return cell_types, true_labels, label_mapping

def read_label_file(label_file_path, label_column_name):  
    try:
        labels_df = pd.read_csv(label_file_path)
        
        if label_column_name and label_column_name in labels_df.columns:
            original_labels = labels_df[label_column_name].values
            return process_labels_unified(original_labels, f"label_file_{label_column_name}")
        else:
            possible_label_columns = ['label', 'cell_type', 'celltype', 'type', 'cluster', 'class', 'annotation']
            label_column_found = None
            for col in possible_label_columns:
                if col in labels_df.columns:
                    label_column_found = col
                    break
            if label_column_found:
                original_labels = labels_df[label_column_found].values
                return process_labels_unified(original_labels, f"auto_found_{label_column_found}")
            else:
                first_col = labels_df.columns[0]
                original_labels = labels_df[first_col].values
                return process_labels_unified(original_labels, f"first_column_{first_col}")
                
    except Exception as e:
        return None, None, None

file_extension = MATRIX_FILE_PATH.lower().split('.')[-1]

labels = None
cell_type_strings = None
original_labels = None
h5_internal_labels = None  

if file_extension == "csv":
    adata = pd.read_csv(MATRIX_FILE_PATH, header=0, index_col=0)
    adata = adata.T
    data = sc.AnnData(adata, dtype=np.float32)

    
elif file_extension == "txt":
    adata = pd.read_csv(MATRIX_FILE_PATH, header=0, index_col=0, delim_whitespace=True)
    adata = adata.T  
    data = sc.AnnData(adata, dtype=np.float32)
    
elif file_extension in ["h5", "hdf5"]:
    with h5py.File(MATRIX_FILE_PATH, 'r') as f:
        print("H5 file structure:")
        for key in f.keys():
            try:
                if hasattr(f[key], 'shape'):
                    print(f"  {key}: shape {f[key].shape}, dtype {f[key].dtype}")
                else:
                    print(f"  {key}: group")
            except:
                print(f"  {key}: (unable to read details)")
        
        X = None
        
        if 'exprs' in f:
            required_fields = ['exprs/data', 'exprs/indices', 'exprs/indptr', 'exprs/shape']
            missing_fields = [field for field in required_fields if field not in f]
            
            data_vals = np.array(f['exprs/data'][:])
            indices = np.array(f['exprs/indices'][:])
            indptr = np.array(f['exprs/indptr'][:])
            shape = tuple(f['exprs/shape'][:])
            
            X = csr_matrix((data_vals, indices, indptr), shape=shape)
            
        elif 'X' in f or 'data' in f:
            if 'X' in f:
                X = np.array(f['X'][:])
            elif 'data' in f:
                X = np.array(f['data'][:])
        
        if 'Y' in f and LABEL_FILE_PATH is None:
            h5_internal_labels = np.array(f['Y'][:])

        
        if X is not None:
            data = sc.AnnData(X, dtype=np.float32)
    


if file_extension in ["h5", "hdf5"] and h5_internal_labels is not None:
    cell_type_strings, labels, label_mapping = process_labels_unified(h5_internal_labels, "h5_internal_Y")
else:
    cell_type_strings, labels, label_mapping = read_label_file(LABEL_FILE_PATH, LABEL_COLUMN_NAME)

if labels is not None and cell_type_strings is not None:
    if len(labels) == data.shape[0]:
        data.obs['label'] = labels
        data.obs['cell_type'] = cell_type_strings


original_shape = data.shape
sc.pp.filter_genes(data, min_cells=3)


raw_counts_file = OUTPUT_DATA_PATH.replace('.csv', '_raw_counts.csv')

raw_counts_df = data.to_df()

raw_counts_df = np.maximum(np.round(raw_counts_df), 0).astype(int)
raw_counts_df.to_csv(raw_counts_file)

sc.pp.normalize_total(data, target_sum=1e4)

sc.pp.log1p(data)


if issparse(data.X):
    if np.any(np.isnan(data.X.data)) or np.any(np.isinf(data.X.data)):
        data.X.data = np.nan_to_num(data.X.data, nan=0.0, posinf=0.0, neginf=0.0)
else:
    if np.any(np.isnan(data.X)) or np.any(np.isinf(data.X)):
        data.X = np.nan_to_num(data.X, nan=0.0, posinf=0.0, neginf=0.0)

if USE_READ_COUNTING:
    data_df = data.to_df()
    data_df = round(data_df)
    data = sc.AnnData(data_df, dtype=np.float32, obs=data.obs)
    sc.pp.highly_variable_genes(data, n_top_genes=784, flavor='seurat_v3', subset=True)
data = data[:, data.var.highly_variable]

hvg_names = data.var_names.tolist()
raw_counts_df = raw_counts_df[hvg_names]
raw_counts_df.to_csv(raw_counts_file)

output_dir = os.path.dirname(OUTPUT_DATA_PATH)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

data_df = data.to_df()
data_df.to_csv(OUTPUT_DATA_PATH)




if 'cell_type' in data.obs.columns and 'label' in data.obs.columns:
    
    string_labels = data.obs['cell_type'].values
    numeric_labels = data.obs['label'].values
    
    if np.min(numeric_labels) != 0:

        unique_nums = sorted(np.unique(numeric_labels))
        remap = {old: new for new, old in enumerate(unique_nums)}
        numeric_labels = np.array([remap[label] for label in numeric_labels])
    
    labels_df = pd.DataFrame({
        'true_label': numeric_labels,   
        'cell_type': string_labels       
    })
    

    output_label_dir = os.path.dirname(OUTPUT_LABEL_PATH)
    if output_label_dir and not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    
    labels_df.to_csv(OUTPUT_LABEL_PATH, index=False)

    
    mapping_file = OUTPUT_LABEL_PATH.replace('.csv', '_mapping.txt')
    unique_types = labels_df.groupby('true_label')['cell_type'].first().sort_index()
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        
        for idx, cell_type in unique_types.items():
            count = (labels_df['true_label'] == idx).sum()
            percentage = count / len(labels_df) * 100