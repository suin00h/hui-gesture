import argparse
import torch
import torch.nn as nn
import numpy as np

from tqdm.auto import tqdm
from torch import optim

from data.dataset import Sign_Language_Dataset
from models.models import DeepConvLSTM
from utils import *

def train(args):
    net = args.net
    opt = args.optimizer
    crit = args.criterion
    device = args.device
    
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    
    dataloader = args.dataloaders[0]
    net.train()
    for batch in dataloader:
        sensor_input = get_concatenated_sensor(args, batch).float().to(device)
        label = batch["label"].long().to(device)
        
        opt.zero_grad()
        
        net_output = net(sensor_input)
        
        loss = crit(net_output, label)
        train_loss.append(loss.item())
        loss.backward()
        opt.step()
        
        equal = (net_output.topk(1)[1].squeeze() == label).float()
        train_accuracy.append(torch.sum(equal))
    
    dataloader = args.dataloaders[1]
    net.eval()
    for batch in dataloader:
        sensor_input = get_concatenated_sensor(args, batch).float().to(device)
        label = batch["label"].long().to(device)
        
        net_output = net(sensor_input)
        
        loss = crit(net_output, label)
        val_loss.append(loss.item())
        
        equal = (net_output.topk(1)[1].squeeze() == label).float()
        val_accuracy.append(torch.sum(equal))
    
    args.loss["train_loss_list"].append(np.mean(train_loss).item())
    args.loss["val_loss_list"].append(np.mean(val_loss).item())
    
    args.accuracy["train_accuracy_list"].append((sum(train_accuracy) / args.dataset_sizes[0]).item())
    args.accuracy["val_accuracy_list"].append((sum(val_accuracy) / args.dataset_sizes[1]).item())
    
    return

def test(args):
    net = args.net
    crit = args.criterion
    device = args.device
    
    test_loss = []
    test_accuracy = []
    
    dataloader = args.dataloaders[2]
    net.eval()
    for batch in dataloader:
        sensor_input = get_concatenated_sensor(args, batch).float().to(device)
        label = batch["label"].long().to(device)
        
        net_output = net(sensor_input)
        
        loss = crit(net_output, label)
        test_loss.append(loss.item())
        
        equal = (net_output.topk(1)[1].squeeze() == label).float()
        test_accuracy.append(torch.sum(equal))
    
    args.loss["test_loss"] = np.mean(test_loss).item()
    args.accuracy["test_accuracy"] = (sum(test_accuracy) / args.dataset_sizes[2]).item()
    
    return

def run(custom_arg=None):
    args = get_parser().parse_args(args=custom_arg)
    get_settings(args)
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = Sign_Language_Dataset()
    dataset_split = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    args.dataset_sizes = [len(split) for split in dataset_split]
    args.dataloaders = [get_dataloader(split, args) for split in dataset_split]
    
    args.net = DeepConvLSTM(args.in_channels, 400, kernel_size=10, stride=2)
    args.optimizer = optim.Adam(args.net.parameters(), lr=args.learning_rate)
    args.criterion = nn.CrossEntropyLoss()
    
    set_metrics(args, ["loss", "accuracy", "f1score"])
    
    args.net.to(args.device)
    for _ in tqdm(range(args.epoch)):
        train(args)
    test(args)
    
    return args

if __name__ == "__main__":
    args = run()    # run([]) for colab environment
    print(args.loss["test_loss"])
