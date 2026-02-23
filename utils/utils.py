import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats


# get basic statistics
def get_statistics(array, title):
    print('===== ' + title + ' =====')
    print(f'min = {np.min(array.reshape(-1)):.2f}')
    print(f'max = {np.max(array.reshape(-1)):.2f}')
    print(f'mean = {np.mean(array.reshape(-1)):.2f}')
    print(f'std = {np.std(array.reshape(-1)):.2f}')
    print(f'Q1 (25%) = {np.quantile(array.reshape(-1), 0.25):.2f}')
    print(f'Q2 (median) = {np.quantile(array.reshape(-1), 0.5):.2f}')
    print(f'Q3 (75%) = {np.quantile(array.reshape(-1), 0.75):.2f}')
    print(f'99% = {np.quantile(array.reshape(-1), 0.99):.2f}')


# apply log normalization
def lognorm(array, base, normalized=False):
    if normalized:
        # default normalization mehtod of the package Seurat
        scaled = 1e6 * array / np.sum(array, axis=0)
    else:
        scaled = array
    
    if base == '2':
        norm = np.log2(1 + scaled)
    elif base == 'e':
        norm = np.log(1 + scaled)

    return np.log(1 + scaled)


# calculate wcss and silhouette score to determine k (# of clusters)
def get_cluster_n(data, k_range, random_seed=0):
    wcss = {}
    scores = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_seed)
        model.fit(data)
        wcss[k] = model.inertia_

        pred = model.predict(data)
        score = silhouette_score(data, pred)
        scores[k] = score

    return wcss, scores


# get cluster labels and print the corresponding samples
def kmeans_result(data, names, n_clusters, random_seed=0):
    model = KMeans(n_clusters=n_clusters, random_state=random_seed)
    cluster_labels = model.fit_predict(data)

    mapped_cluster_name = {cluster: [] for cluster in range(n_clusters)}
    for i in range(len(names)):
        label = cluster_labels[i]
        sample = names[i]
    
        mapped_cluster_name[label].append(sample)
    
    for cluster, sample in mapped_cluster_name.items():
        print ("Cluster: ", cluster, "\n", sample)

    return mapped_cluster_name


# make a csv file storing cluster information
def make_csv_cluster(mapped_cluster_name, file_name):
    n_clusters = len(mapped_cluster_name.keys())
    df = pd.DataFrame(columns=['cluster', 'sample'])

    for cluster in range(n_clusters):
        list_samples = mapped_cluster_name[cluster]
        for sample in list_samples:
            new_row = pd.DataFrame([[cluster, sample]], columns=df.columns)
            df = pd.concat([df, new_row])

    df.to_csv('./Export/CSV/' + file_name)


# transpose dataframe
def transpose_df(df_data):
    df_T = df_data.T
    new_columns = [''] + df_T.columns.values
    df_T.columns = new_columns
    return df_T


# get meta dataframe
def get_df_meta(con_ids, sam_ids):
    label_con = 'control'
    label_sam = 'treated'
    label_con_meta = np.array([[id, label_con] for id in con_ids])
    label_sam_meta = np.array([[id, label_sam] for id in sam_ids])
    label_meta = np.concatenate([label_con_meta, label_sam_meta])

    df_meta = pd.DataFrame(label_meta, columns=['','condition'])
    df_meta = df_meta.set_index('')
    return df_meta


# DESeq inference

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import math

def do_DESeq_inference(df_transposed, df_meta, plot=False, stress='', log2fc=1, padj=0.001):
    inference = DefaultInference(n_cpus=32)

    dds = DeseqDataSet(
        counts=df_transposed,
        metadata=df_meta,
        design_factors="condition",
        # contrast="condition",
        refit_cooks=True,
        inference=inference)

    
    dds.deseq2()
    stat_res = DeseqStats(dds, inference=inference, contrast=["condition", "treated", "control"])
    stat_res.summary()
    df_stat_res = stat_res.results_df
    down = df_stat_res[(df_stat_res['log2FoldChange']<=-1*(log2fc)) & (df_stat_res['padj']<=padj)]
    up = df_stat_res[(df_stat_res['log2FoldChange']>=log2fc) & (df_stat_res['padj']<=padj)]

    if plot:
        plt.scatter(x=df_stat_res['log2FoldChange'], y=df_stat_res['padj'].apply(lambda x: -np.log10(x)), s=1, label="Not significant", color="grey")
        plt.scatter(x=down['log2FoldChange'], y=down['padj'].apply(lambda x:-np.log10(x)), s=3, label="Down-regulated", color="green")
        plt.scatter(x=up['log2FoldChange'], y=up['padj'].apply(lambda x: -np.log10(x)), s=3, label="Up-regulated", color="red")

        plt.xlabel("log2FC")
        plt.ylabel("-logFDR")
        plt.title(stress)
        plt.axvline(-1*log2fc, color="grey", linestyle="--")
        plt.axvline(log2fc, color="grey", linestyle="--")
        plt.axhline(-1*math.log10(padj), color="grey", linestyle="--")
        plt.show()
        plt.close()

    df_stat_res.to_csv(f"Export/{stress}_DEG.csv", index=True)

    DEG_up = list(up.index)
    DEG_down = list(down.index)
    DEG = list(up.index) + list(down.index)
    print(f"Number of DEG Up: {len(DEG_up)} | Number of DEG Down: {len(DEG_down)} | Total number of DEG: {len(DEG)}")

    return DEG, DEG_up, DEG_down


# update confusion matrix
def update_cm(confusion, true_y, pred_y):
    rows = true_y.cpu().numpy() 
    cols = pred_y.max(1)[1].detach().cpu().numpy()
    for row, col in zip(rows, cols):
        confusion[row, col] += 1
    return confusion


# get f1-score from a confusion matrix
def get_f1(cm, epsilon=1e-9):
    # adding epsilon to 0 diagonal values
    idx0 = np.where(cm.diagonal()==0)[0]
    for i in idx0:
        cm[i, i] += epsilon

    precision = cm.diagonal() / cm.sum(axis=1)
    recall = cm.diagonal() / cm.sum(axis=0)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return np.nanmean(f1)


def list_to_txt(id_list, out_name):
    with open(out_name, 'w') as out_file:
        for id in id_list:
            out_file.write(id.strip() + "\n")


def min_max_norm_TPM(TPM_file, min_max_dic={}):
    print("\nMin-Max scaling of input data is in progress..\n")
    # TPM_df=pd.read_csv(TPM_file, index_col=0)

    TPM_df = TPM_file

    index_val = TPM_df.index
    column_val = TPM_df.columns
    TPM_np_max1_sorted = TPM_df.values
    
    tpm_data_trans = TPM_np_max1_sorted.T
    tpm_data_trans_log2 = np.log2(tpm_data_trans + 1)
    TPM_np_max1_sorted = tpm_data_trans_log2.T
    
    norm_tpm_list=[]

    if min_max_dic == {}:
        min_max_dic = {}
        for posi, data in enumerate(TPM_np_max1_sorted):
            gene_ID = index_val[posi]

            max_val = max(data)
            min_val = min(data)
            if max_val != 0 and max_val != min_val:
                norm_data = (data - min_val) / (max_val - min_val)
            else:
                norm_data = data
            norm_tpm_list.append(norm_data)

            min_max_dic[gene_ID] = {'min':float(min_val), 'max':float(max_val)}

        norm_tpm_df = pd.DataFrame(norm_tpm_list, index=index_val, columns=column_val)
        return norm_tpm_df, min_max_dic
    
    else:
        for posi, data in enumerate(TPM_np_max1_sorted):
            gene_ID = index_val[posi]

            max_val = float(min_max_dic[gene_ID]['max'])
            min_val = float(min_max_dic[gene_ID]['min'])
            if max_val != 0 and max_val != min_val:
                norm_data = (data - min_val) / (max_val - min_val)
            else:
                norm_data = data
            norm_tpm_list.append(norm_data)

        norm_tpm_df = pd.DataFrame(norm_tpm_list, index=index_val, columns=column_val)

        return norm_tpm_df, min_max_dic



