import argparse
import torch

from sklearn import metrics

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
    parser.add_argument("--dataset-sizes")
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
    parser.add_argument("--loss")
    parser.add_argument("--accuracy")
    parser.add_argument("--f1score")
    
    
    return parser

def get_settings(args):
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
    
    return

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

def show_settings():
    for i, setting in enumerate(SETTINGS):
        print(f"{i}: {setting['sensor_type']}")
    return

def run_epoch(args, phase: str):
    no_train = phase_idx = ["train", "val", "test"].index(phase)
    dataloaders = args.dataloaders[phase_idx]
    net = args.net
    opt = args.optimizer
    device = args.device
    
    loss_list = []
    acc_list = []
    f1_list = []
    
    if no_train:
        net.eval()
    else:
        net.train()
    
    for batch in dataloaders:
        sensor_input = [batch[sensor].float() for sensor in args.in_sensors]
        if not args.concat_latent:
            sensor_input = torch.cat(sensor_input, dim=2)
        label = batch["label"].long().to(device)
        
        if not no_train: opt.zero_grad()
        net_output = net(sensor_input)
        
        loss = args.criterion(net_output, label)
        loss_list.append(loss.item())
        if not no_train:
            loss.backward()
            opt.step()
        
        true_positive = (net_output.topk(1)[1].squeeze() == label).float()
        acc_list.append(torch.sum(true_positive))
        
        f1score = metrics.f1_score(
            torch.argmax(net_output, dim=1).cpu().numpy(),
            label.cpu().numpy(),
            average="weighted"
        )
        f1_list.append(f1score)
    
    return loss_list, acc_list, f1_list
