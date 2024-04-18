""" 
    Implementation of Training 
"""

import argparse
import yaml
import datetime
import time
import torch
import pickle
from torch.utils.data import DataLoader

from dataset.OIADataset import BDDOIA
from model.GetModel import get_model
from utils.TrainingUtils import create_lr_scheduler, train_one_epoch, evaluate


def main(args):
    """ Main Function """
    # config
    config_file = f"./{args.dataset}_config.yaml"
    with open(f"{config_file}", 'r') as f:
        config = yaml.safe_load(f)
    
    # device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # log file
    now_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = f"./log/log_{now_time}.txt"

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
    dataset_train = BDDOIA(imageRoot = config["data"]["bddoia_data"],
                           actionRoot = config["data"]["train_action"],
                           reasonRoot = config["data"]["train_reason"],
                    )
    dataset_val = BDDOIA(imageRoot = config["data"]["bddoia_data"],
                         actionRoot = config["data"]["val_action"],
                         reasonRoot = config["data"]["val_reason"],

                    )
    dataloader_train = DataLoader(dataset_train,
                                batch_size=config["optimizer"]["batch_size"], 
                                shuffle=True, 
                                pin_memory=True,
                                drop_last=True,
                                num_workers=4)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=config["optimizer"]["batch_size"], 
                                pin_memory=True,
                                drop_last=True,
                                num_workers=4)
    # get model
    model = get_model(config)
    model.to(device)
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.cbam.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
        {"params": [p for p in model.neck.parameters() if p.requires_grad]},
        {"params": [p for p in model.attention_module.parameters() if p.requires_grad]},
        {"params": [p for p in model.gnn.parameters() if p.requires_grad]},
        {"params": [p for p in model.common_classifier_head.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier_head_action.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier_head_reason.parameters() if p.requires_grad]},
    ]

    # optimizer
    optimizer = torch.optim.SGD(params_to_optimize, 
                                lr=config["optimizer"]["learning_rate"], 
                                momentum=config["optimizer"]["momentum"], 
                                weight_decay=config["optimizer"]["weight_decay"],
                            )
    
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    epochs = config["optimizer"]["num_epoches"]
    iterations = len(dataloader_train)
    lr_scheduler = create_lr_scheduler(optimizer, iterations, epochs, warmup=True)

    # training
    start_time = time.time()
    for epoch in range(epochs):
        total_loss, lr = train_one_epoch(model, 
                                        optimizer, 
                                        dataloader_train,
                                        label_embedding,
                                        adj_COO,
                                        edge_attr,
                                        weight_tensor,
                                        device, 
                                        epoch,
                                        lr_scheduler=lr_scheduler, 
                                        print_freq=args.print_freq, 
                                        scaler=scaler
                                    )
        # total_loss, lr = 0, 0
        val_res = evaluate(model, dataloader_val, label_embedding, adj_COO, edge_attr, weight_tensor, device)

        # log results
        train_info = f"[epoch: {epoch}]\n" \
                     f"train_loss: {total_loss:.4f}\n" \
                     f"lr: {lr:.6f}\n"
        val_info = f'Val_loss: {val_res["Val_loss"]}\n' \
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
            f.write(train_info + val_info + f"total_metric: {total_metric} \n\n\n")

        # save model
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, f"./save_model/model_{epoch}.pth")

    total_time = time.time() - start_time
    total_time = datetime.timedelta(seconds=int(total_time))
    print(f"Finish training with {total_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--dataset", default="bddoia", type=str, help="Dataset")
    parser.add_argument("--amp", default=True, type=bool, help="Whether Use torch.cuda.amp")
    parser.add_argument('--print_freq', default=100, type=int, help='Print frequency')
    args = parser.parse_args()
    
    main(args)
    
    
