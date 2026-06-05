import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
import os
from collections import Counter
                                                                                   
from NB_NEW_sc import HybridSCCAD, HybridDeepClusterWithMetrics, NBLoss
from NB_NEW_network import AutoEncoder, myBottleneck
from Metrics import nmi, acc, ari

os.environ['OMP_NUM_THREADS'] = '6'

class read_Data(Dataset):
    def __init__(self, norm_data, raw_counts_data=None, labels=None, transform=None):
        self.transform = transform
        self.norm_data = self.load_data(norm_data, labels)
        self.raw_counts_data = self.load_data(raw_counts_data, labels) if raw_counts_data is not None else None

    def load_data(self, data, labels):
        data = np.array(data).astype('float32')
        
        B = []
        for i in range(len(data)):
            t = data[i, :]
            t = t.reshape((28, 28))
            B.append(t)
        B = np.array(B)

        if labels is not None:
            if hasattr(labels, 'values'):
                if hasattr(labels, 'columns') and 'true_label' in labels.columns:
                    label_values = labels['true_label'].values
                    print("   >>> Using 'true_label' column (numeric labels)")
                else:
                    label_values = labels.iloc[:, 0].values if hasattr(labels, 'iloc') else labels.values
                    print("   >>> Using first column as labels")
            else:
                label_values = labels

            label_values = np.array(label_values)
            
            if label_values.ndim > 1:
                label_values = label_values.flatten()
            
            try:
                labee = label_values.astype(int).tolist()
            except ValueError as e:
                raise ValueError(
                    f"Labels must be numeric! Please ensure 'true_label' column contains integers. "
                    f"Error: {e}"
                )
        else:
            labee = [0] * len(data)

        data_list = []
        for i in range(len(data)):
            data_list.append((B[i], labee[i]))
        
        return data_list

    def __len__(self):
        return len(self.norm_data)

    def __getitem__(self, index):
        norm_image_info, img_label = self.norm_data[index]
        
        if self.raw_counts_data is not None:
            raw_image_info, _ = self.raw_counts_data[index]
            if self.transform:
                norm_sample = self.transform(norm_image_info)
                raw_sample = self.transform(raw_image_info)
            return norm_sample, raw_sample, img_label, index

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

def main():
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    
    data_path = "/home/jinrongqi/scSMD-3-NB_NB 2D/data/Darmanis/"
    results_path = "/home/jinrongqi/scSMD-3-NB_NB 2D/results/Darmanis/"
    os.makedirs(results_path, exist_ok=True)
    
    required_files = ["processed_data.csv", "celltypes.csv"]
    optional_files = ["processed_data_raw_counts.csv"]
    
    for file in required_files:
        if not os.path.exists(data_path + file):
            raise FileNotFoundError(f"Required file not found: {data_path + file}")
    
    csv_label = pd.read_csv(data_path + "celltypes.csv", header=0, index_col=None)
    
    if 'true_label' not in csv_label.columns:
        raise ValueError("celltypes.csv must contain 'true_label' column with numeric labels")
    
    csv_label_numeric = csv_label[['true_label']].copy()
    
    unique_labels = csv_label_numeric['true_label'].unique()
    num_cluster = len(unique_labels)
    
    label_counts = csv_label_numeric['true_label'].value_counts().sort_index()
    
    if 'cell_type' in csv_label.columns:
        print(">>> Numeric label -> Cell type mapping:")
        for label in sorted(unique_labels):
            cell_type = csv_label[csv_label['true_label'] == label]['cell_type'].iloc[0]
            count = (csv_label_numeric['true_label'] == label).sum()
    
    data = pd.read_csv(data_path + "processed_data.csv", header=0, index_col=0)
    

    if len(csv_label) != data.shape[0]:
        raise ValueError(f"Label count ({len(csv_label)}) does not match cell count ({data.shape[0]})")
    
    raw_counts_data = None
    raw_counts_file = data_path + "processed_data_raw_counts.csv"
    if os.path.exists(raw_counts_file):
        raw_counts_data = pd.read_csv(raw_counts_file, header=0, index_col=0)

    batch_size = 100
    dataset_size = data.shape[0]
    transform_fn = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = read_Data(data, raw_counts_data, csv_label, transform=transform_fn)
    kwargs = {'num_workers': 1}
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True, 
        drop_last=False, 
        **kwargs
    )
    pretraining_epoch = 20
    T1 = 2
    MaxIter1 = 20
    m = 1.5
    latent_size = 10
    dataset_name = 'rare_cell'
    a = 0.2



    try:
        AE = AutoEncoder(myBottleneck, [1, 1, 1]).cuda()
        AE.apply(weights_init)
    except Exception as e:
        print(f"Error initializing network: {e}")
        return None

    try:
        deep_cluster = HybridDeepClusterWithMetrics(
            AE, 
            data_loader, 
            dataset_size, 
            batch_size=batch_size,
            pretraining_epoch=pretraining_epoch,
            MaxIter1=MaxIter1, 
            num_cluster=num_cluster, 
            m=m, 
            T1=T1,
            latent_size=latent_size, 
            dataset_name=dataset_name, 
            a=a
        )
    except Exception as e:
        print(f"Error creating deep clustering model: {e}")
        return None

    ARI_list, NMI_list, ACC_list = [], [], []
    
    if pretraining_epoch != 0:
        try:
            deep_cluster.pretrain()
        except Exception as e:
            return None
    if MaxIter1 != 0:
        try:
            ARI_list, NMI_list, ACC_list = deep_cluster.first_module()
            
        except Exception as e:
            return None
    try:
        deep_cluster_labels = deep_cluster.get_final_clustering_result()
        unique_clusters = len(np.unique(deep_cluster_labels))
    except Exception as e:
        print(f"Error getting clustering results: {e}")
        return None
    
    try:
        hybrid_model = HybridSCCAD()

        X_norm = data.values.astype('float32')
        cellNames = data.index.tolist()
        geneNames = data.columns.tolist()
        dataName = dataset_name  
        
        result, score, sub_clusters, degs_list = hybrid_model.hybrid_scCAD(
            X_norm=X_norm,                           
            cellNames=cellNames,                     
            geneNames=geneNames,                     
            deep_init_clusters=deep_cluster_labels,  
            dataName=dataName,
            save_path=results_path,
            merge_h=50,                              
            overlap_h=0.7,                           
            rare_h=0.01,                            
            seed=2023,
            save_full=True 
        )
    except Exception as e:
        print(f"Error during rare cell detection: {e}")
        return None


    has_true_labels = False
    cell_to_label = {}
    overall_label_distribution = {}
    
    try:
        if os.path.exists(data_path + "celltypes.csv"):
            label_df = pd.read_csv(data_path + "celltypes.csv", header=0, index_col=None)
            true_labels = label_df.iloc[:, 0]
            
            if len(true_labels) == len(cellNames):
                has_true_labels = True
                cell_to_label = dict(zip(cellNames, true_labels))
                overall_label_distribution = Counter(true_labels)
                for cell_type, count in overall_label_distribution.most_common():
                    percentage = (count / len(cellNames)) * 100

    except Exception as e:
        print(f"error: {e}")
    
    saved_files = []
    expected_files = [
        f"ARI_{dataset_name}.csv",
        f"NMI_{dataset_name}.csv", 
        f"ACC_{dataset_name}.csv",
        f"{dataset_name}_clusters.csv",
        f"{dataset_name}_rare_cells_result.txt",
        f"{dataset_name}_degs_list.txt",
        f"{dataset_name}_comb_sub-clusters.txt",
        f"{dataset_name}_sub-clusters_anomaly_score.txt"
    ]
    
    for file in expected_files:
        if os.path.exists(results_path + file):
            saved_files.append(file)
            print(f"   ✓ {file}")
    
    
    return {
        'clustering_metrics': {
            'ARI': ARI_list,
            'NMI': NMI_list,
            'ACC': ACC_list
        },
        'rare_cell_results': {
            'rare_clusters': result,
            'anomaly_scores': score,
            'sub_clusters': sub_clusters,
            'degs_list': degs_list
        },
        'deep_cluster_labels': deep_cluster_labels,
        'summary': {
            'total_cells': len(cellNames),
            'total_genes': len(geneNames),
            'rare_clusters_found': len(result),
            'total_rare_cells': sum(len(cluster) for cluster in result) if result else 0,
            'deep_clustering_clusters': len(np.unique(deep_cluster_labels)),
            'final_subclusters': len(np.unique(sub_clusters)) if len(sub_clusters) > 0 else 0,
            'best_ARI': max(ARI_list) if ARI_list else 0,
            'best_NMI': max(NMI_list) if NMI_list else 0,
            'best_ACC': max(ACC_list) if ACC_list else 0,
            'used_raw_counts': raw_counts_data is not None,
            'num_clusters_detected': num_cluster
        }
    }

if __name__ == '__main__':
    results = main()