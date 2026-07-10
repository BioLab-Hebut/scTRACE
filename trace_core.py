import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from collections import Counter
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import warnings
import time
import networkx as nx
from community import community_louvain

from metrics import nmi, acc, ari


class NegativeBinomialLoss(nn.Module):
    def __init__(self):
        super(NegativeBinomialLoss, self).__init__()

    def forward(self, x, mean, disp):
        eps = 1e-10
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2
        return nb_final


class TraceClusterTrainer(nn.Module):
    def __init__(self, AE, data_loader, dataset_size, batch_size=100, 
                 pretraining_epoch=20, MaxIter1=20, num_cluster=8, 
                 m=1.5, T1=2, latent_size=10, dataset_name='hybrid', a=0.1):
        super(TraceClusterTrainer, self).__init__()
        self.AE = AE
        self.u_mean = torch.zeros([num_cluster, latent_size])
        self.batch_size = batch_size
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter1 = MaxIter1
        self.num_cluster = num_cluster
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.m = m
        self.T1 = T1
        self.latent_size = latent_size
        self.dataset_name = dataset_name
        self.a = a

    def _unpack_batch_data(self, batch_data):
        if len(batch_data) != 4:
            raise ValueError("Expected batch format: norm_x, raw_x, target, index")
        return batch_data

    def evaluate_clustering(self):
        self.AE.eval()
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        
        for batch_data in self.data_loader:
            norm_x, raw_x, target, index = self._unpack_batch_data(batch_data)
            
            norm_x = Variable(norm_x).cuda()
            _mean, _disp, _pi, u, y = self.AE(norm_x)
            
            u = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
            p = torch.zeros([min(self.batch_size, norm_x.shape[0]), self.num_cluster]).cuda()
            for j in range(self.num_cluster):
                p[:, j] = torch.sum(torch.pow(
                    u[j, :, :] - self.u_mean[j, :].unsqueeze(0).repeat(min(self.batch_size, norm_x.shape[0]), 1), 2),
                                    dim=1)
            p = torch.pow(p, -1 / (self.m - 1))
            sum1 = torch.sum(p, dim=1)
            p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
            y = torch.argmax(p, dim=1)
            
            y = y.cpu().numpy()
            pred_labels[ii * min(self.batch_size, norm_x.shape[0]):(ii + 1) * min(self.batch_size, norm_x.shape[0])] = y
            true_labels[ii * min(self.batch_size, norm_x.shape[0]):(ii + 1) * min(self.batch_size, norm_x.shape[0])] = target.numpy()
            ii += 1

        NMI = nmi(true_labels, pred_labels)
        ARI = ari(true_labels, pred_labels)
        ACC = acc(true_labels, pred_labels, self.num_cluster)
        
        
        self.AE.train()
        return NMI, ARI, ACC

    def update_cluster_centers(self):
        self.AE.eval()
        for param in self.AE.parameters():
            param.requires_grad = False
        den = torch.zeros([self.num_cluster]).cuda()
        num = torch.zeros([self.num_cluster, self.latent_size]).cuda()
        
        for batch_data in self.data_loader:
            norm_x, raw_x, target, index = self._unpack_batch_data(batch_data)
            
            norm_x = Variable(norm_x).cuda()
            _mean, _disp, _pi, u, y = self.AE(norm_x)

            p = torch.zeros([min(self.batch_size, norm_x.shape[0]), self.num_cluster]).cuda()
            for j in range(0, self.num_cluster):
                p[:, j] = torch.sum(torch.pow(
                    u.unsqueeze(0).repeat(self.num_cluster, 1, 1)[j, :, :] - self.u_mean[j, :].unsqueeze(0).repeat(
                        min(self.batch_size, norm_x.shape[0]), 1), 2), dim=1)
            p = torch.pow(p, -1 / (self.m - 1))
            sum1 = torch.sum(p, dim=1)
            p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))

            p = torch.pow(p, self.m)
            for kk in range(0, self.num_cluster):
                den[kk] = den[kk] + torch.sum(p[:, kk])
                num[kk, :] = num[kk, :] + torch.matmul(p[:, kk].t(), u)
                
        for kk in range(0, self.num_cluster):
            self.u_mean[kk, :] = torch.div(num[kk, :], den[kk])
            
        self.AE.cuda()
        self.AE.train()
        for param in self.AE.parameters():
            param.requires_grad = True
        return self.u_mean

    def pretrain(self):
        print(">>> Starting pretraining with MSE + NB loss...")
        self.AE.train()
        self.AE.cuda()
        for param in self.AE.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(self.AE.parameters())
        
        print(">>> Using RAW COUNTS data for NB loss calculation")

        for T in range(self.pretraining_epoch):
            print('Pretraining Iteration: ', T + 1)
            total_loss = 0
            total_mse_loss = 0
            total_negative_binomial_loss = 0
            
            for batch_data in self.data_loader:
                norm_x, raw_x, target, index = self._unpack_batch_data(batch_data)
                
                optimizer.zero_grad()
                norm_x = Variable(norm_x).cuda()
                raw_x = Variable(raw_x).cuda()
                
                _mean, _disp, _pi, u, y = self.AE(norm_x)
                
                mse_loss = nn.MSELoss()(norm_x, y)
                
                nb_input = raw_x.view(-1, 784)
                
                negative_binomial_loss_val = torch.mean(
                    torch.sum(self.negative_binomial_loss(nb_input, _mean, _disp), dim=1)
                )
                
                nb_weight = 0.001
                loss = mse_loss + nb_weight * negative_binomial_loss_val
                loss = mse_loss

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_mse_loss += mse_loss.item()
                total_negative_binomial_loss += negative_binomial_loss_val.item()
            
            print(f'Average Loss: {total_loss/len(self.data_loader):.4f} '
                  f'(MSE: {total_mse_loss/len(self.data_loader):.4f}, '
                  f'NB: {total_negative_binomial_loss/len(self.data_loader):.4f})')

        print(">>> Pretraining completed with MSE + NB loss.")
        return self.AE

    def initialize_cluster_centers(self):
        print(">>> Initializing cluster centers...")
        self.AE.cuda()
        
        datas = np.zeros([self.dataset_size, self.latent_size])
        ii = 0
        for batch_data in self.data_loader:
            norm_x, raw_x, target, index = self._unpack_batch_data(batch_data)
            
            norm_x = Variable(norm_x).cuda()
            _mean, _disp, _pi, u, y = self.AE(norm_x)
            u = u.cpu()
            datas[ii * min(self.batch_size, norm_x.shape[0]):(ii + 1) * min(self.batch_size, norm_x.shape[0])] = u.data.numpy()
            ii = ii + 1

        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)
        self.u_mean = kmeans.cluster_centers_
        self.u_mean = torch.from_numpy(self.u_mean)
        self.u_mean = Variable(self.u_mean).cuda()
        return self.AE, self.u_mean

    def cluster_objective(self, x, y, u, p, u_means):
        return torch.matmul(p, torch.sum(torch.pow(x - y, 2), dim=1)) + self.a * torch.matmul(p, torch.sum(      
            torch.pow(u - u_means, 2), dim=1))     

    def negative_binomial_loss(self, x, mean, disp):
        negative_binomial_loss = NegativeBinomialLoss().cuda()
        return negative_binomial_loss(x, mean, disp)

    def train_deep_clustering(self):
        print(">>> Starting First Module Training...")
        self.AE, self.u_mean = self.initialize_cluster_centers()
        self.AE.cuda()
        self.AE.train()
        for param in self.AE.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(self.AE.parameters(), lr=0.00001)
        
        print(">>> Using RAW COUNTS data for NB loss calculation")

        ARIlist = []
        NMIlist = []
        ACClist = []
        
        for T in range(self.MaxIter1):
            print(f'First Module Iteration: {T + 1}/{self.MaxIter1}')
            if T % self.T1 == 1:
                self.u_mean = self.update_cluster_centers()
                
            for batch_data in self.data_loader:
                norm_x, raw_x, target, index = self._unpack_batch_data(batch_data)
                
                u = torch.zeros([self.num_cluster, min(self.batch_size, norm_x.shape[0]), self.latent_size]).cuda()
                norm_x = Variable(norm_x).cuda()
                raw_x = Variable(raw_x).cuda()
                
                for kk in range(self.num_cluster):
                    _mean, _disp, _pi, u1, y = self.AE(norm_x)
                    u[kk, :, :] = u1.cuda()
                u = u.detach()

                p = torch.zeros([min(self.batch_size, norm_x.shape[0]), self.num_cluster]).cuda()
                for j in range(self.num_cluster):
                    p[:, j] = torch.sum(torch.pow(
                        u[j, :, :] - self.u_mean.cuda()[j, :].unsqueeze(0).repeat(min(self.batch_size, norm_x.shape[0]), 1),
                        2), dim=1)
                p = torch.pow(p, -1 / (self.m - 1))
                sum1 = torch.sum(p, dim=1)
                p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))

                p = p.detach()
                self.u_mean = self.u_mean.cuda()
                p = p.T
                p = torch.pow(p, self.m)
                
                for i in range(self.num_cluster):
                    _mean, _disp, _pi, u1, y = self.AE(norm_x)
                    self.u_mean = self.u_mean.float()
                    

                    loss1 = self.cluster_objective(norm_x.view(-1, 784), y.view(-1, 784), u1, p[i, :].unsqueeze(0),
                                                 self.u_mean[i, :].unsqueeze(0).repeat(min(self.batch_size, norm_x.shape[0]), 1))
                    
                    _mean = _mean.float()
                    _disp = _disp.float()
                    
                    nb_input = raw_x.view(-1, 784)

                    loss2 = torch.matmul(p[i, :].unsqueeze(0),
                                         torch.sum(self.negative_binomial_loss(nb_input, _mean, _disp), dim=1))

                    beta = 0.1
                    loss = loss1 + beta * loss2

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            NMI, ARI, ACC = self.evaluate_clustering()
            ARIlist.append(ARI)
            NMIlist.append(NMI)
            ACClist.append(ACC)
        
        print(f">>> First module training completed!")
        print(f">>> Final metrics - NMI: {NMIlist[-1]:.4f}, ARI: {ARIlist[-1]:.4f}, ACC: {ACClist[-1]:.4f}")
        return ARIlist, NMIlist, ACClist

    def predict_clusters(self):
        print(">>> Getting final clustering results...")
        
        self.AE.eval()
        self.AE.cuda()
        self.u_mean = self.u_mean.cuda()
        
        predict_list = []
        cell_index = []

        for batch_data in self.data_loader:
            norm_x, raw_x, target, index = self._unpack_batch_data(batch_data)
            
            norm_x = Variable(norm_x).cuda(non_blocking=True)
            _mean, _disp, _pi, u, y = self.AE(norm_x)
            
            p = torch.zeros([norm_x.shape[0], self.num_cluster]).cuda()
            for j in range(self.num_cluster):
                p[:, j] = torch.sum(torch.pow(
                    u - self.u_mean[j, :].unsqueeze(0).repeat(norm_x.shape[0], 1), 2), dim=1)
            p = torch.pow(p, -1 / (self.m - 1))
            sum1 = torch.sum(p, dim=1)
            p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
            
            y = torch.argmax(p, dim=1)
            y = y.cpu().numpy()

            for i in range(norm_x.shape[0]):
                p_idx = index.numpy()[i]
                cell_index.append(p_idx)
                predict_list.append(y[i])
        
        sorted_result = sorted(zip(cell_index, predict_list))
        final_cluster_labels = [label for _, label in sorted_result]
        
        return np.array(final_cluster_labels)


def write_nested_list(lst, filename):
    with open(filename, 'w') as file:
        for sublist in lst:
            line = '\t'.join(str(element) for element in sublist)
            file.write(line + '\n')


class TraceDetector:
    def __init__(self):
        self.deep_cluster = None
        
    def fast_clustering(self, data, k=15, seed=2023):
        nps = min(40, data.shape[0])
        if data.shape[0] >= nps:
            pca = PCA(n_components=nps, random_state=seed)
            X_pca = pca.fit_transform(data)
        else:
            X_pca = data

        from sklearn.neighbors import NearestNeighbors
        nn_res = NearestNeighbors(n_neighbors=(k+1), algorithm='auto', n_jobs=-1).fit(X_pca)
        distances, indices = nn_res.kneighbors(X_pca)

        G = nx.Graph()
        for i in range(indices.shape[0]):
            for j in indices[i][1:]:
                G.add_edge(i, j)

        partition = community_louvain.best_partition(G, random_state=seed)
        return partition

    def detect_rare_cells(self, csv_file_path=None, X_norm=None, cellNames=None, geneNames=None,
                     deep_init_clusters=None, dataName=None, seed=2023, merge_h=50, 
                     overlap_h=0.7, rare_h=0.01, save_full=True, save_path='./'):
        warnings.filterwarnings("ignore")
        if dataName is None:
            dataName = "rare_cell"

        print(">>> scTRACE algorithm starting...")
        
        if X_norm is None:
            if csv_file_path is None:
                raise ValueError("Must provide either csv_file_path or X_norm")
            print(f">>> Loading preprocessed data from: {csv_file_path}")
            data_df = pd.read_csv(csv_file_path, header=0, index_col=0)
            X_norm = data_df.values.astype('float32')
            cellNames = data_df.index.tolist()
            geneNames = data_df.columns.tolist()
        
        if X_norm.shape[1] != 784:
            raise ValueError(f"Expected 784 features after preprocessing, got {X_norm.shape[1]}")
        
        n_cells = X_norm.shape[0]
        n_genes = X_norm.shape[1]
        print(f">>> Loaded data - Cells: {n_cells}; Genes: {n_genes}")
        
        start = time.time()

        print(">>> Clusters decomposition using traditional methods...")
        pseudo_init_subclusters = deep_init_clusters.copy()
        pseudo_subclusters = pseudo_init_subclusters.copy()
        h1 = max(int(rare_h * n_cells), 30)
        count = 1

        while 1:
            print(">>> iter %d, running..." % count)
            count = count + 1
            
            dict_count = Counter(pseudo_subclusters)
            depths = {}
            dpt = 1
            
            while dict_count.most_common(1)[0][1] >= h1:
                depths.update(
                    (key, dpt) for key in list(set([i for i, count in dict_count.items() if count < h1]) - set(depths.keys())))
                dpt = dpt + 1
                c_max = max(pseudo_subclusters) + 1
                c_list = list(set([i for i, count in dict_count.items() if count >= h1]) - set(depths.keys()))
                if len(c_list) == 0:
                    break
                
                for clustid in c_list:
                    idx = np.where(pseudo_subclusters == clustid)[0]
                    temp_X = X_norm[idx, :].copy()
                    temp_clusters = self.fast_clustering(data=temp_X, seed=seed)
                    temp_clusters = [temp_clusters[node] + c_max for node in range(temp_X.shape[0])]
                    if len(np.unique(temp_clusters)) != 1:
                        pseudo_subclusters[idx] = temp_clusters
                        c_max = max(pseudo_subclusters) + 1
                    else:
                        depths[clustid] = dpt - 1
                        
                dict_count = Counter(pseudo_subclusters)

            subc_dict = {}
            p = 0
            rename_pseudo_subclusters = np.zeros(n_cells, dtype=int)
            for i in range(n_cells):
                if pseudo_subclusters[i] in subc_dict.keys():
                    rename_pseudo_subclusters[i] = subc_dict[pseudo_subclusters[i]]
                else:
                    subc_dict[pseudo_subclusters[i]] = p
                    rename_pseudo_subclusters[i] = p
                    p = p + 1
            n_subclusters = len(np.unique(rename_pseudo_subclusters))
            print(">>> After clusters decomposition, we got %d balanced sub-clusters." % (n_subclusters))

            print(">>> Clusters merge in progress...")
            X_centers = np.zeros((len(np.unique(rename_pseudo_subclusters)), X_norm.shape[1]))
            for i in np.unique(rename_pseudo_subclusters):
                id = np.where(rename_pseudo_subclusters == i)[0]
                X_centers[i, :] = np.mean(X_norm[id, :], axis=0)

            distances = pdist(X_centers)
            distance_matrix = squareform(distances)
            np.fill_diagonal(distance_matrix, np.max(distance_matrix) + 1)
            comb_id = []
            for i in range(X_centers.shape[0]):
                i_nearest_neighbor_index = np.argsort(distance_matrix[i, :])
                for j in i_nearest_neighbor_index:
                    dist = distance_matrix[i, j]
                    if dist <= np.percentile(np.min(distance_matrix, axis=0), merge_h) and sorted([i, j]) not in comb_id:
                        comb_id.append(sorted([i, j]))
            merged = []
            for sublist in comb_id:
                merged_with = None
                for m in merged:
                    if any(x in m for x in sublist):
                        merged_with = m
                        break
                if merged_with:
                    merged.remove(merged_with)
                    merged.append(list(set(merged_with + sublist)))
                else:
                    merged.append(sublist)
            comb_subclusters = rename_pseudo_subclusters.copy()
            for i in tqdm(range(len(merged))):
                for j in range(n_cells):
                    if comb_subclusters[j] in merged[i]:
                        comb_subclusters[j] = min(merged[i])
            subc_dict = {}
            p = 0
            rename_comb_subclusters = np.zeros(n_cells, dtype=int)
            for i in range(n_cells):
                if comb_subclusters[i] in subc_dict.keys():
                    rename_comb_subclusters[i] = subc_dict[comb_subclusters[i]]
                else:
                    subc_dict[comb_subclusters[i]] = p
                    rename_comb_subclusters[i] = p
                    p = p + 1
            n_subclusters = len(np.unique(comb_subclusters))
            print(">>> After clusters merge, we got %d sub-clusters." % (n_subclusters))
            result_df = pd.DataFrame(rename_comb_subclusters, columns=['Cluster_Label'])
            csv_file_path = save_path + dataName + '_final_clustered_result.csv'
            result_df.to_csv(csv_file_path, index=False)
            print(f"Final merged cluster result saved to {csv_file_path}")

            print(">>> Cluster anomaly score calculation in progress...")
            IFmodel = IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
            overlap = []
            degs_list = []
            for i in tqdm(range(n_subclusters)):
                id = np.where(rename_comb_subclusters == i)[0]
                if len(id) > h1 or len(id) < 10:
                    overlap.append(0)
                    degs_list.append([])
                else:
                    id_ = np.where(rename_comb_subclusters != i)[0]
                    tmp_X = X_norm[id, :]
                    zero_cols = np.where(np.all(tmp_X == 0, axis=0))[0]
                    re_cols = list(set(np.arange(X_norm.shape[1])) - set(zero_cols))
                    
                    if len(re_cols) == 0:
                        overlap.append(0)
                        degs_list.append([])
                        continue
                        
                    tmp_X = X_norm[:, re_cols]
                    diff = np.abs(np.median(tmp_X[id, :], axis=0) - np.median(tmp_X[id_, :], axis=0))
                    var_names = np.array(geneNames)[re_cols]
                    n_top = min(20, len(np.where(diff > 0)[0]))
                    
                    if n_top == 0:
                        overlap.append(0)
                        degs_list.append([])
                        continue
                        
                    degs_ = list(var_names[np.argsort(-diff)[:n_top]])
                    degs_list.append(degs_)
                    
                    selected_cols = [re_cols[j] for j in np.argsort(-diff)[:n_top]]
                    IFmodel.fit(X_norm[:, selected_cols])
                    s = IFmodel.score_samples(X_norm[:, selected_cols])
                    overlap.append(len(set(np.argsort(s)[:len(id)]) & set(id)) / len(id))

            remain_clusters = []
            overlap = np.array(overlap)
            remain_degs_list = []
            for i in range(n_subclusters):
                if overlap[i] >= overlap_h:
                    remain_clusters.append(i)
                    remain_degs_list.append(degs_list[i])
            if len(remain_clusters) != 0:
                break
            elif rare_h > 0.05:
                print(">>> Rare type (<5%) not found!")
                break
            else:
                if 30 >= int(rare_h * n_cells):
                    rare_h = 30/n_cells + 0.01
                else:
                    rare_h = rare_h + 0.01
                h1 = int(rare_h * n_cells)
                pseudo_subclusters = pseudo_init_subclusters.copy()
        end = time.time()
        runningtime = end - start
        print(">>> time used:", runningtime)

        result = []
        for i in remain_clusters:
            id = np.where(rename_comb_subclusters == i)[0]
            if cellNames is None:
                result.append(list(id))
            else:
                result.append(list(np.array(cellNames)[id]))
        if save_full:
            np.savetxt(save_path + dataName + '_trace_balanced_sub-clusters.txt', rename_pseudo_subclusters, fmt='%d')
            np.savetxt(save_path + dataName + '_trace_comb_sub-clusters.txt', rename_comb_subclusters, fmt='%d')
            np.savetxt(save_path + dataName + '_trace_sub-clusters_anomaly_score.txt', overlap, fmt='%f')
        write_nested_list(result, save_path + dataName + '_trace_rare_cells_result.txt')
        write_nested_list(remain_degs_list, save_path + dataName + '_trace_degs_list.txt')

        print(f">>> scTRACE completed!")
        print(f">>> Found {len(result)} rare cell clusters")
        if len(result) > 0:
            for i, cluster in enumerate(result):
                print(f">>> Rare cluster {i+1}: {len(cluster)} cells")
        print(f">>> Final rare_h threshold used: {rare_h:.4f}")
        return result, overlap, rename_comb_subclusters, remain_degs_list