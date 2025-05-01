import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data.dataset import SIGN_LABELS
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedKFold, train_test_split

SETTINGS = [
    dict(
        sensor_type = "acceleration",
        in_channels = 3,
        sensor_key = ["imu_acc"]
    ),
    dict(
        sensor_type = "gyroscope",
        in_channels = 3,
        sensor_key = ["imu_gyro"]
    ),
    dict(
        sensor_type = "orientation",
        in_channels = 4,
        sensor_key = ["imu_ori"]
    ),
    dict(
        sensor_type = "EMG",
        in_channels = 8,
        sensor_key = ["emg"]
    )
]

def get_parser():
    parser = argparse.ArgumentParser()
    
    # Experiment hyperparameters
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    
    # Training modules
    parser.add_argument("--net")
    parser.add_argument("--device")
    parser.add_argument("--optimizer")
    parser.add_argument("--criterion")
    parser.add_argument("--dataloaders")
    
    # Experiment settings
    parser.add_argument("--setting-description")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--kfold", default=5, type=int)
    parser.add_argument("--save-log", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--sensor-idxs", action="extend", nargs="+", type=int)
    parser.add_argument("--concat-latent", action="store_true")
    
    # Model parameters
    parser.add_argument("--in-sensors")
    parser.add_argument("--in-channels")
    parser.add_argument("--input-length", default=400, type=int)
    parser.add_argument("--kernel-size", default=10, type=int)
    parser.add_argument("--stride", default=2, type=int)
    parser.add_argument("--lstm-hidden-size", default=128, type=int)
    parser.add_argument("--lstm-layers", default=2, type=int)
    parser.add_argument("--num-class", default=26, type=int)
    
    return parser

def set_settings(args):
    setting_description = "Using "
    in_channels = []
    in_sensors = []
    
    for sensor_idx in args.sensor_idxs:
        sensor_dict = SETTINGS[sensor_idx]
        
        setting_description += sensor_dict["sensor_type"] + " "
        in_channels.append(sensor_dict["in_channels"])
        in_sensors += sensor_dict["sensor_key"]
    
    args.setting_description = setting_description
    args.in_channels = in_channels
    args.in_sensors = in_sensors

def set_device(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

def get_kfold_dataloaders(dataset, labels, seed, k, batch_size):
    kfold_list = []
    
    train_val_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.1,
        stratify=labels,
        random_state=seed
    )
    
    train_val_labels = labels[train_val_idx]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    for train_idx, val_idx in skf.split(train_val_idx, train_val_labels):
        train_dataset = Subset(dataset, train_val_idx[train_idx])
        val_dataset = Subset(dataset, train_val_idx[val_idx])
        kfold_list.append([
            get_dataloader(train_dataset, batch_size),
            get_dataloader(val_dataset, batch_size)
        ])
    
    test_dataloader = get_dataloader(Subset(dataset, test_idx), batch_size)
    
    return kfold_list, test_dataloader

def set_dataloader(args, dataloaders):
    dataset_zip = zip(["train", "val", "test"], dataloaders)
    args.dataloaders = {phase: get_dataloader(args, split) for phase, split in dataset_zip}

def get_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

def get_metric_dict():
    return dict(
        train=[],
        val=[],
        test=[]
    )

def show_settings():
    for i, setting in enumerate(SETTINGS):
        print(f"{i}: {setting['sensor_type']}")

def plot_metrics(train_metrics, val_metrics, test_metric, title, figsize=(6, 6), ylim=(0, 1)):
    print(f"Test {title}: {test_metric[0]:.2f}")
    plt.figure(figsize=figsize)
    plt.plot(train_metrics, label="Train")
    plt.plot(val_metrics, label="Validation")

    for metric in [train_metrics, val_metrics]:
        last_value = metric[-1]
        plt.annotate(
            f"{last_value:.2f}",
            xy=(len(metric) - 1, last_value), xytext=(0, 15),
            textcoords="offset points",
            ha="center", fontsize=10, color="red",
            arrowprops=dict(arrowstyle="->", color="red")
        )
    
    plt.xlabel("Epoch")
    plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()

def show_confusion_matrix(confusion_matrix, figsize=(6, 6)):
    annot = np.empty_like(confusion_matrix, dtype=object)
    for row in range(confusion_matrix.shape[0]):
      total = sum(confusion_matrix[row])
      for col in range(confusion_matrix.shape[1]):
        if confusion_matrix[row, col]:
          annot[row, col] = f"{confusion_matrix[row, col] / total:.2f}\n({int(confusion_matrix[row, col])})"
        else:
          annot[row, col] = '0'
    
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix, cmap="Blues", annot=annot, fmt="s", xticklabels=SIGN_LABELS, yticklabels=SIGN_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.yticks(rotation=0)
    
    plt.show()
