""" 
    Implementation of Training Utils
"""
import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.MetricLogger import MetricLogger, SmoothedValue


def criterion(output1, output2, output3, target1, target2, target3, device):
    """ Loss Function """
    losses = {}
    class_weights1 = [1, 1, 2, 2]
    w1 = torch.FloatTensor(class_weights1).to(device)

    losses['action'] = torch.nn.functional.binary_cross_entropy_with_logits(output1, target1, weight=w1)
    losses['reason'] = torch.nn.functional.binary_cross_entropy_with_logits(output2, target2)
    losses['relation'] = torch.nn.functional.mse_loss(output3[:143], target3)

    return 0.5 * losses['action'] + 0.75 * losses['reason'] + 0.15 * losses['relation']


def create_lr_scheduler(optimizer, 
                        num_step, 
                        epochs, 
                        warmup=True, 
                        warmup_epochs=1, 
                        warmup_factor=1e-3
                    ):
    """ Learning Rate Update Strategy """
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def List2Arr(List):
    """ Convert List to Array """
    Arr1 = np.array(List[:-1]).reshape(-1, List[0].shape[1])
    Arr2 = np.array(List[-1]).reshape(-1, List[0].shape[1])

    return np.vstack((Arr1, Arr2))


def train_one_epoch(model, 
                    optimizer, 
                    dataloader_train,
                    label_embedding,
                    adj_COO, 
                    edge_attr,
                    weight_tensor,
                    device, 
                    epoch,
                    lr_scheduler,
                    print_freq, 
                    scaler
                    ):
    """ Training One Epoch"""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target, _ in metric_logger.log_every(dataloader_train, print_freq, header):
        image = image.to(device)

        target[0] = target[0].to(device)
        target[1] = target[1].to(device)
    
        with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image, label_embedding, adj_COO, edge_attr)
                output1 = output[0].to(device)
                output2 = output[1].to(device)
                output3 = output[2].to(device)
                loss = criterion(output1, output2, output3, target[0], target[1], weight_tensor, device)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].globals, lr


def evaluate(model, data_loader, label_embedding, adj_COO, edge_attr, weight_tensor, device):
    """ Get Evaluation Results """
    model.eval()
    with torch.no_grad():
        val_loss = 0
        Target_ActionArr = []
        Pre_ActionArr = []
        Target_ReasonArr = []
        Pre_ReasonArr = []

        for image, target, _ in tqdm(data_loader):
            image = image.to(device)
            output = model(image, label_embedding, adj_COO, edge_attr)
            output1 = output[0]
            output2 = output[1]
            output3 = output[2]
            target[0] = target[0].to(device)
            target[1] = target[1].to(device)
            # loss
            loss = criterion(output1, output2, output3, target[0], target[1], weight_tensor, device)
            val_loss += loss

            # calculate the F1 score
            predict_action = torch.sigmoid(output1) > 0.5
            preds_action = predict_action.cpu().numpy()
            predict_reason = torch.sigmoid(output2) > 0.5
            preds_reason = predict_reason.cpu().numpy()

            a_targets = target[0].cpu().numpy()
            e_targets = target[1].cpu().numpy()

            Target_ActionArr.append(a_targets)
            Pre_ActionArr.append(preds_action)
            Target_ReasonArr.append(e_targets)
            Pre_ReasonArr.append(preds_reason)

        Target_ActionArr = List2Arr(Target_ActionArr)
        Pre_ActionArr = List2Arr(Pre_ActionArr)
        Target_ReasonArr = List2Arr(Target_ReasonArr)
        Pre_ReasonArr = List2Arr(Pre_ReasonArr)

        # action
        Action_F1_overall = f1_score(Target_ActionArr, Pre_ActionArr, average='samples')
        Action_Per_action = f1_score(Target_ActionArr, Pre_ActionArr, average=None)
        Action_F1_mean = np.mean(Action_Per_action)

        # reason
        Reason_F1_overall = f1_score(Target_ReasonArr, Pre_ReasonArr, average='samples')
        Reason_Per_action = f1_score(Target_ReasonArr, Pre_ReasonArr, average=None)
        Reason_F1_mean = np.mean(Reason_Per_action)

    res_dict = {}
    res_dict["Val_loss"] = val_loss.item()
    res_dict["Action_overall"] = np.mean(Action_F1_overall)
    res_dict["Reason_overall"] = np.mean(Reason_F1_overall)
    res_dict["F1_action"] = Action_Per_action
    res_dict["F1_action_average"] = Action_F1_mean
    res_dict["F1_reason"] = Reason_Per_action
    res_dict["F1_reason_average"] = Reason_F1_mean

    return res_dict
