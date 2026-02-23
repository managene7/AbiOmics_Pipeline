import numpy as np
import pandas as pd
import torch


def get_data(file_path, index_column=''):
    df = pd.read_csv(file_path)
    if len(index_column) > 0:
        df = df.set_index(index_column)
    return df


def txt_to_list(file_path):
    with open(file_path, 'r') as f:
        file_lines = f.readlines()
    out_list=[f.strip() for f in file_lines if f !=""]
    return out_list


def dataloader(data, label, random_state=1):
    rng = np.random.default_rng(random_state)
    rand_index = rng.permutation(len(data))

    x_data = np.array(data)
    y_data = np.array(label)

    x_data = torch.FloatTensor(x_data[rand_index])
    y_data = torch.LongTensor(y_data[rand_index])

    return x_data, y_data

