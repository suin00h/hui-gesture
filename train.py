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
    train_loss, train_accuracy, train_f1score = run_epoch(args, phase="train")
    val_loss, val_accuracy, val_f1score = run_epoch(args, phase="val")
    
    args.loss["train_loss_list"].append(np.mean(train_loss).item())
    args.loss["val_loss_list"].append(np.mean(val_loss).item())
    
    args.accuracy["train_accuracy_list"].append((sum(train_accuracy) / args.dataset_sizes[0]).item())
    args.accuracy["val_accuracy_list"].append((sum(val_accuracy) / args.dataset_sizes[1]).item())
    
    args.f1score["train_f1score_list"].append(np.mean(train_f1score).item())
    args.f1score["val_f1score_list"].append(np.mean(val_f1score).item())
    
    return

def test(args):
    test_loss, test_accuracy, test_f1score = run_epoch(args, phase="test")
    
    args.loss["test_loss"] = np.mean(test_loss).item()
    args.accuracy["test_accuracy"] = (sum(test_accuracy) / args.dataset_sizes[2]).item()
    args.f1score["test_f1score"] = np.mean(test_f1score).item()
    
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
    args = run()    # run([]) or run("{custom arguments}") for colab environment
    print(args.f1score["test_f1score"])
    print(args.f1score["train_f1score_list"])
