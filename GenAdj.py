"""
    Implementation of Generate Adjacent Matrix
"""
import yaml
import pickle
import torch
import numpy as np
from tqdm import tqdm

from dataset.OIADataset import BDDOIA


def normalization_matrix(co_matrix, num_action_class, num_reason_class):
    """ Normalization Matrix from Two Dim """
    assert co_matrix.shape[0] == (num_action_class + num_reason_class)
    assert np.allclose(co_matrix, co_matrix.T)
    for idx, row_array in enumerate(co_matrix):
        sum1 = sum(row_array[:num_action_class])
        sum2 = sum(row_array[num_action_class:])
        co_matrix[idx][:num_action_class] /= sum1
        co_matrix[idx][num_action_class:] /= sum2
    return co_matrix


def binary_matrix(co_matrix, num_action_class, num_reason_class, action_threshold, reason_threshold):
    """ Binary Matrix to Avoid Overfit """
    assert co_matrix.shape[0] == (num_action_class + num_reason_class)
    for idx, row_array in enumerate(co_matrix):
        for jdx in range(len(row_array)):
            if jdx < num_reason_class:
                if co_matrix[idx][jdx] >= action_threshold:
                    co_matrix[idx][jdx] = 1
                else:
                    co_matrix[idx][jdx] = 0
            else:
                if co_matrix[idx][jdx] >= reason_threshold:
                    co_matrix[idx][jdx] = 1
                else:
                    co_matrix[idx][jdx] = 0
    return co_matrix


def ratio_matrix(co_matrix, num_action_class, num_reason_class, ratio=0.5, min_ratio=0.1):
    """ Keep the Sumation of Others being a Pre-defined Ratio to Avoid Oversmooth """
    assert co_matrix.shape[0] == (num_action_class + num_reason_class)
    for idx, row_array in enumerate(co_matrix):
        sum1 = sum(row_array[:num_action_class])
        sum2 = sum(row_array[num_action_class:])
        if sum1 != 0:
            for jdx in range(num_reason_class):
                if co_matrix[idx][jdx] == 1:
                    co_matrix[idx][jdx] = ratio / sum1
        if sum2 != 0:
            for jdx in range(num_reason_class, num_action_class + num_reason_class):
                if co_matrix[idx][jdx] == 1:
                    co_matrix[idx][jdx] = ratio / sum2

    for idx, row_array in enumerate(co_matrix):
        for jdx in range(len(row_array)):
            if co_matrix[idx][jdx] != 0 and co_matrix[jdx][idx] == 0:
                co_matrix[jdx][idx] = min_ratio

    return co_matrix


def numpy2tensor(co_matrix, num_action_class, num_reason_class):
    """ Conver Co_matrix Numpy to Tensor """
    res_list = [[], []]
    weight_list = []
    arr_list = []
    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            if co_matrix[i][j] > 0:
                res_list[0].append(i)
                res_list[1].append(j)
                weight_list.append(round(co_matrix[i][j], 2))
            
                if i < num_action_class:
                    if j < num_action_class:
                        arr_list.append([0, 0])
                    else:
                        arr_list.append([0, 1])
                else:
                    if j < num_action_class:
                        arr_list.append([1, 0])
                    else:
                        arr_list.append([1, 1])
    weight_list += [1] * (num_reason_class + num_action_class)

    assert len(res_list[0]) == len(res_list[1])
    assert len(res_list[0]) == len(arr_list)
    
    adj_COO = torch.tensor(res_list, dtype=torch.int64)
    weight_tensor = torch.tensor(weight_list, dtype=torch.float32)
    arr_tensor = torch.tensor(arr_list, dtype=torch.int64)
    res = {
        "weight_tensor": weight_tensor,
        "adj_COO": adj_COO,
        "arr_tensor": arr_tensor
    }
    return res


if __name__ == "__main__":
    # config
    config_file = "./bddoia_config.yaml"
    with open(f"{config_file}", 'r') as f:
        CONFIG = yaml.safe_load(f)

    dataset_train = BDDOIA(imageRoot = CONFIG["data"]["bddoia_data"],
                           actionRoot = CONFIG["data"]["train_action"],
                           reasonRoot = CONFIG["data"]["train_reason"],
                       )
    # initail co_matrix
    num_action_class = CONFIG["action_class"]
    num_reason_class = CONFIG["reason_class"]
    num_of_col = num_action_class + num_reason_class
    co_matrix = np.zeros(shape=[num_of_col, num_of_col])
    
    # count co-occurrence in training set
    for img, targets, _ in tqdm(dataset_train):
        action_labels = targets[0].tolist()
        reason_labels = targets[1].tolist()
        # co-occurrence
        index_action = [idx for idx, value in enumerate(action_labels) if value == 1.0]
        index_reason = [(idx + num_action_class) for idx, value in enumerate(reason_labels) if value == 1.0]
        index_all = index_action + index_reason
        # modify co_matrix
        for i in index_all:
            for j in index_all:
                if i == j:
                    continue
                co_matrix[i][j] += 1
                co_matrix[j][i] += 1

    # normalization
    co_matrix = normalization_matrix(co_matrix, num_action_class, num_reason_class)

    # avoid overfit
    action_threshold = CONFIG["GNN"]["action_threshold"]
    reason_threshold = CONFIG["GNN"]["reason_threshold"]
    co_matrix = binary_matrix(co_matrix, num_action_class, num_reason_class, action_threshold, reason_threshold)

    # avoid oversmooth
    ratio = CONFIG["GNN"]["ratio"]
    min_ratio = CONFIG["GNN"]["min_ratio"]
    co_matrix = ratio_matrix(co_matrix, num_action_class, num_reason_class, ratio=ratio, min_ratio=min_ratio)

    # convert to tensor
    res = numpy2tensor(co_matrix, num_action_class, num_reason_class)

    # store adj COO to file
    file_name = CONFIG["adj_file_path"]
    with open(f"{file_name}", "wb") as f:
        pickle.dump(res, f)
    print(f"Finish get adjacency information and storing it in {file_name}")
