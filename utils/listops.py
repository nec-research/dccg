import torch
import json
import numpy as np


def json2data(json_file, vocab_dict):
    with open(json_file, 'r') as file:
        data_json = json.load(file)
    data_dict = {}
    for size_s, dataset in data_json.items():
        size = int(size_s)
        T_ex, T_t, T_ad, T_i, T_im = list(), list(), list(), list(), list()
        for data in dataset.values():
            ex = [vocab_dict[tok] for tok in data['ex'].split(' ')]
            T_ex.append(torch.LongTensor(ex))
            T_t.append(torch.tensor(data['target']))
            mat = torch.zeros(size, size, dtype=torch.int64)
            for i, idx in enumerate(data['adj']):
                if i == 0:
                    continue
                mat[i, idx] = 1
            T_ad.append(mat)
            inter = list()
            for x in data['inter']:
                inte = x + 4 if x > -1 else 0
                inter.append(inte)
            T_i.append(torch.LongTensor(inter))
            T_im.append(torch.LongTensor(data['inter_mask']))
        T_ex = torch.cat([t.unsqueeze(0) for t in T_ex], 0)
        T_t = torch.cat([t.unsqueeze(0) for t in T_t], 0)
        T_ad = torch.cat([t.unsqueeze(0) for t in T_ad], 0)
        T_i = torch.cat([t.unsqueeze(0) for t in T_i], 0)
        T_im = torch.cat([t.unsqueeze(0) for t in T_im], 0)
        T_data = torch.utils.data.TensorDataset(T_ex, T_t, T_ad, T_i, T_im)
        data_dict[size] = T_data
    return data_dict


def get_dataloader(dataset_dict, batch_size, shuffle=True):
    data_loader = list()
    num_examples = 0
    for dataset in dataset_dict.values():
        data, target, A, inter, inter_mask = dataset[:]
        num = data.size(0)
        shuffle_idx = torch.randperm(num) if shuffle else torch.arange(num)
        for start in range(0, num, batch_size):
            idx = shuffle_idx[start:start+batch_size]
            batch = (data[idx], target[idx], A[idx],
                     inter[idx], inter_mask[idx])
            data_loader.append(batch)
            num_examples += idx.shape[0]
    np.random.shuffle(data_loader) if shuffle else data_loader
    return data_loader, num_examples
