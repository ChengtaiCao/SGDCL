""" 
    Implementation of Inference 
"""

import argparse
import yaml
import datetime
import time
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.OIADataset import BDDOIA
from model.GetModel import get_model
from utils.TrainingUtils import evaluate


def main(args):
    """ Main Function """
    import numpy as np
    import random
    # config
    config_file = f"./{args.dataset}_config.yaml"
    with open(f"{config_file}", 'r') as f:
        config = yaml.safe_load(f)
    
    # device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # log file
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = f"./log/TestResults_{now_time}.txt"

    # label embedding
    word_embedding_path = config["word_embedding_path"]
    with open(f"{word_embedding_path}", "rb") as f:
        label_embedding = pickle.load(f).to(device)

    # adj information
    adj_file_path = config["adj_file_path"]
    with open(f"{adj_file_path}", "rb") as f:
        adj_info = pickle.load(f)
        edge_attr = adj_info["arr_tensor"].to(device)
        adj_COO = adj_info["adj_COO"].to(device)
        weight_tensor = adj_info["weight_tensor"].to(device)

    # prepare data
    dataset_test = BDDOIA(imageRoot = config["data"]["bddoia_data"],
                         actionRoot = config["data"]["test_action"],
                         reasonRoot = config["data"]["test_reason"],

                    )
    dataloader_test = DataLoader(dataset_test,
                                batch_size=10, 
                                pin_memory=True,
                                drop_last=False,
                                num_workers=4)
    # get model
    model = get_model(config, pretrained=False)
    model.to(device)

    start_time = time.time()
    epochs = config["optimizer"]["num_epoches"]
    max_result_index = 0
    max_result = -float("inf")
    for epoch in range(epochs):
        print(epoch)

        checkpoint = torch.load(f"./save_model/model_{epoch}.pth", map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        val_res = evaluate(model, dataloader_test, label_embedding, adj_COO, edge_attr, weight_tensor, device)

        # log results
        val_info = f"[epoch: {epoch}]\n" \
                   f'Val_loss: {val_res["Val_loss"]}\n' \
                   f'Action_overall: {val_res["Action_overall"]}\n' \
                   f'Reason_overall: {val_res["Reason_overall"]}\n' \
                   f'F1_action: {val_res["F1_action"]}\n' \
                   f'F1_action_average: {val_res["F1_action_average"]}\n' \
                   f'F1_reason: {val_res["F1_reason"]}\n' \
                   f'F1_reason_average: {val_res["F1_reason_average"]}\n'
        
        total_metric = float(val_res["Action_overall"]) + \
                       float(val_res["Reason_overall"]) + \
                       float(val_res["F1_action_average"]) + \
                       float(val_res["F1_reason_average"])

        with open(log_file, "a") as f:
            f.write(val_info + f"total_metric: {total_metric} \n\n\n")
        
        if total_metric > max_result:
            max_result = total_metric
            max_result_index = epoch

    total_time = time.time() - start_time
    total_time = datetime.timedelta(seconds=int(total_time))
    print(f"Finish testing with {total_time}")
    print(f"Max_result is in {max_result_index} and total metric is {max_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--dataset", default="bddoia", type=str, help="Dataset")
    parser.add_argument("--amp", default=True, type=bool, help="Whether Use torch.cuda.amp")
    parser.add_argument('--print_freq', default=20, type=int, help='Print frequency')
    args = parser.parse_args()
    
    main(args)
    
    
