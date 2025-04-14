import torch
import numpy as np

from torch import optim
from sklearn import metrics
from tqdm.auto import tqdm

from utils import *
from data.dataset import Sign_Language_Dataset

def run(custom_arg=None):
    parser = get_parser()
    args = parser.parse_args(args=custom_arg)
    
    set_settings(args)
    set_device(args)
    set_dataloader(args, Sign_Language_Dataset(), [0.8, 0.1, 0.1])
    set_network(args)
    set_metrics(args)
    
    args.optimizer = optim.Adam(args.net.parameters(), lr=args.learning_rate)
    args.criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(args)
    trainer.run()
    
    return args

class Trainer():
    def __init__(self, args):
        self.net = args.net
        self.device = args.device
        self.test_only = args.test_only
        self.epoch = args.epoch
        self.metrics = args.metrics
        self.dataloaders = args.dataloaders
        self.in_sensors = args.in_sensors
        self.concat_latent = args.concat_latent
        self.opt = args.optimizer
        self.crit = args.criterion
    
    def run(self):
        self.network_to_device()
        
        if not self.test_only:
            self.train_and_val()
        self.run_phase("test")
    
    def run_phase(self, phase: str):
        metrics = self.run_epoch(phase)
        
        self.log_metrics(metrics, phase)
    
    def run_epoch(self, phase: str):
        is_train = phase == "train"
        
        metric_epoch = dict(
            loss=[],
            accuracy=[],
            f1score=[]
        )
        
        if is_train:
            self.net.train()
        else:
            self.net.eval()
        
        for batch in self.dataloaders[phase]:
            sensor_input, label = self.get_inputs_from_batch(batch)
            
            if is_train:
                self.opt.zero_grad()
            
            net_output = self.net(sensor_input)
            
            loss = self.crit(net_output, label)
            metric_epoch["loss"].append(loss.item())
            
            if is_train:
                loss.backward()
                self.opt.step()
            
            accuracy = self.get_accuracy(net_output, label)
            metric_epoch["accuracy"].append(accuracy)
            
            f1score = self.get_f1score(net_output, label)
            metric_epoch["f1score"].append(f1score)
        
        return metric_epoch
    
    def network_to_device(self):
        self.net.to(self.device)
    
    def train_and_val(self):
        for _ in tqdm(range(self.epoch)):
            self.run_phase("train")
            self.run_phase("val")
    
    def get_inputs_from_batch(self, batch):
        sensor_input = self.get_sensor_list(batch)
        if not self.concat_latent:
            sensor_input = torch.cat(sensor_input, dim=2)
        label = batch["label"].long().to(self.device)
        
        return sensor_input, label
    
    def get_sensor_list(self, batch):
        return [batch[sensor].float().to(self.device) for sensor in self.in_sensors]
    
    def get_accuracy(self, output, label):
        true_positive = output.topk(1)[1].squeeze() == label
        return true_positive.sum().item() / len(label)
    
    def get_f1score(self, output, label):
        return metrics.f1_score(
            torch.argmax(output, dim=1).cpu().numpy(),
            label.cpu().numpy(),
            average="weighted"
        )
    
    def log_metrics(self, metrics, phase):
        for key, value in metrics.items():
            metric = np.mean(value).item()
            self.metrics[key][phase].append(metric)

if __name__ == "__main__":
    args = run()    # run([]) or run("{custom arguments}") for colab environment
    print(args.metrics["loss"]["train"])
    print(args.metrics["loss"]["val"])
    print(args.metrics["loss"]["test"])
    print(args.metrics["f1score"]["train"])
    print(args.metrics["f1score"]["val"])
    print(args.metrics["f1score"]["test"])
    print(args.metrics["accuracy"]["train"])
    print(args.metrics["accuracy"]["val"])
    print(args.metrics["accuracy"]["test"])
