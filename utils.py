import argparse
import torch

SETTINGS = [
    dict(
        setting_name = "Only acceleration",
        in_channels = 3,
        in_sensors = ["imu_acc"]
    ),
    dict(
        setting_name = "Only gyroscope",
        in_channels = 3,
        in_sensors = ["imu_gyro"]
    ),
    dict(
        setting_name = "Only orientation",
        in_channels = 4,
        in_sensors = ["imu_ori"]
    ),
    dict(
        setting_name = "Only EMG",
        in_channels = 8,
        in_sensors = ["emg"]
    ),
    dict(
        setting_name = "All sensor w/ input level concatenation",
        in_channels = 3 + 3 + 4 + 8,
        in_sensors = ["imu_acc", "imu_gyro", "imu_ori", "emg"]
    )
]

def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preset-idx", default=0, type=int)
    parser.add_argument("--epoch", default=200, type=int)
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--learning-rate", default=2e-4, type=float)
    parser.add_argument("--test-only", action="store_true")
    
    parser.add_argument("--device")
    parser.add_argument("--dataset-sizes")
    parser.add_argument("--dataloaders")
    
    parser.add_argument("--net")
    parser.add_argument("--optimizer")
    parser.add_argument("--criterion")
    
    parser.add_argument("--loss")
    parser.add_argument("--accuracy")
    parser.add_argument("--f1score")
    
    parser.add_argument("--setting-name")
    parser.add_argument("--in-channels")
    parser.add_argument("--in-sensors")
    
    return parser

def get_settings(args):
    preset_dict = SETTINGS[args.preset_idx]
    args.setting_name = preset_dict["setting_name"]
    args.in_channels = preset_dict["in_channels"]
    args.in_sensors = preset_dict["in_sensors"]
    
    return

def get_concatenated_sensor(args, batch):
    batch_list = [batch[sensor] for sensor in args.in_sensors]
    return torch.cat(batch_list, dim=2)

def get_dataloader(dataset, args):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

def set_metrics(args, metrics):
    for metric in metrics:
        setattr(args, metric, {
            f"train_{metric}_list": [],
            f"val_{metric}_list": [],
            f"test_{metric}": None
        })
    return
