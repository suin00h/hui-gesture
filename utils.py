import argparse
import torch

from models.models import *

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
    
    # Train settings
    parser.add_argument("--sensor-idxs", action="extend", nargs="+", type=int)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--concat-latent", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    
    parser.add_argument("--setting-title")
    
    # Hyperparameters
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    
    # Internal variables
    ## Data
    parser.add_argument("--dataloaders")
    
    ## Model
    parser.add_argument("--device")
    parser.add_argument("--net")
    parser.add_argument("--optimizer")
    parser.add_argument("--criterion")
    
    parser.add_argument("--in-channels")
    parser.add_argument("--in-sensors")
    parser.add_argument("--input-length", default=400)
    parser.add_argument("--kernel-size", default=10)
    parser.add_argument("--stride", default=2)
    parser.add_argument("--lstm-hidden-size", default=128)
    parser.add_argument("--lstm-layers", default=2)
    parser.add_argument("--num-classes", default=26)
    
    ## Metrics
    parser.add_argument("--metrics")
    
    
    return parser

def set_settings(args):
    setting_title = "Using "
    in_channels = []
    in_sensors = []
    
    for sensor_idx in args.sensor_idxs:
        sensor_dict = SETTINGS[sensor_idx]
        
        setting_title += sensor_dict["sensor_type"] + " "
        in_channels.append(sensor_dict["in_channels"])
        in_sensors += sensor_dict["sensor_key"]
    
    args.setting_title = setting_title
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

def set_network(args):
    if args.concat_latent:
        args.net = DeepConvLSTM_latent(args)
    else:
        args.net = DeepConvLSTM(args)

def set_metrics(args):
    args.metrics = dict(
        loss=get_metric_dict(),
        accuracy=get_metric_dict(),
        f1score=get_metric_dict(),
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
