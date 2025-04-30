import argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt

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
    parser.add_argument("--log-dir")
    parser.add_argument("--setting-description")
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

def set_dataloader(args, dataset, lengths):
    dataset_split = torch.utils.data.random_split(dataset, lengths)
    dataset_zip = zip(["train", "val", "test"], dataset_split)
    args.dataloaders = {phase: get_dataloader(args, split) for phase, split in dataset_zip}

def get_dataloader(args, dataset):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
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

def show_confusion_matrix(confusion_matrix, figsize=(6, 6)):
    cm = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, cmap="Blues", annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    
    plt.show()
