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
    args.trainer = trainer
    
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
        self.confusion_matrix = self.set_confusion_matrix(args.num_classes)
    
    def run(self):
        self.network_to_device()
        
        if not self.test_only:
            self.train_and_val()
        self.run_phase("test")
    
    def run_phase(self, phase: str):
        phase_metrics = self.run_epoch(phase)
        
        self.log_metrics(phase_metrics, phase)
    
    def run_epoch(self, phase: str):
        is_train = phase == "train"
        self.net.train() if is_train else self.net.eval()
        
        metrics_epoch = { "loss": [], "accuracy": [], "f1score": [] }
        
        for batch in self.dataloaders[phase]:
            sensor_input, label = self.get_inputs_from_batch(batch)
            
            if is_train:
                net_output, loss = self.train_step(sensor_input, label)
            else:
                net_output, loss = self.eval_step(sensor_input, label)
            
            accuracy = self.get_accuracy(net_output, label)
            f1score = self.get_f1score(net_output, label)
            if phase == "test":
                self.confusion_matrix += self.get_confusion_matrix(net_output, label)
            
            metrics_epoch["loss"].append(loss.item())
            metrics_epoch["accuracy"].append(accuracy)
            metrics_epoch["f1score"].append(f1score)
        
        return metrics_epoch
    
    def train_step(self, sensor_input, label):
        self.opt.zero_grad()
        net_output, loss = self.compute_loss(sensor_input, label)
        
        loss.backward()
        self.opt.step()
        
        return net_output, loss
    
    def eval_step(self, sensor_input, label):
        with torch.no_grad():
            return self.compute_loss(sensor_input, label)
    
    def compute_loss(self, sensor_input, label):
        net_output = self.net(sensor_input)
        loss = self.crit(net_output, label)
        
        return net_output, loss
    
    def network_to_device(self):
        self.net.to(self.device)
    
    def train_and_val(self):
        for _ in tqdm(range(self.epoch)):
            self.run_phase("train")
            self.run_phase("val")
    
    def get_inputs_from_batch(self, batch):
        sensor_input = self.get_sensor_list(batch)
        label = batch["label"].long().to(self.device)
        
        return sensor_input, label
    
    def get_sensor_list(self, batch):
        return [batch[sensor].float().to(self.device) for sensor in self.in_sensors]
    
    def get_accuracy(self, output, label):
        true_positive = torch.argmax(output, dim=1).squeeze() == label
        return true_positive.sum().item() / len(label)
    
    def get_f1score(self, output, label):
        return metrics.f1_score(
            torch.argmax(output, dim=1).cpu().numpy(),
            label.cpu().numpy(),
            average="macro"
        )
    
    def log_metrics(self, metrics, phase):
        for key, value in metrics.items():
            metric = np.mean(value).item()
            self.metrics[key][phase].append(metric)
    
    def set_confusion_matrix(self, num_classes):
        return np.zeros((num_classes, num_classes))

    def get_confusion_matrix(self, output, true):
        pred = torch.argmax(output, dim=1).squeeze()
        return metrics.confusion_matrix(true, pred, labels=np.arange(26))

def test_code(code_idx):
    if code_idx == 1:
        args = run()    # run([]) or run("{custom arguments}") for colab environment
        print(args.metrics["loss"]["train"])
        print(args.metrics["loss"]["test"])
        print(args.metrics["f1score"]["test"])
        print(args.metrics["accuracy"]["test"])
    elif code_idx == 2:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        y_true = np.random.randint(0, 5, (10,))
        y_pred = np.random.randint(0, 5, (10,))
        
        cm = metrics.confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, ax=axes[0, 0], cmap="Blues")
        print(y_true, y_pred, cm)
        
        y_true = np.random.randint(0, 5, (10,))
        y_pred = np.random.randint(0, 5, (10,))
        
        cm2 = metrics.confusion_matrix(y_true, y_pred)
        sns.heatmap(cm2, ax=axes[0, 1], cmap="Blues")
        sns.heatmap(cm + cm2, ax=axes[1, 0], cmap="Blues")
        
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        print(y_true)
        print(y_pred)
        plt.show()
    elif code_idx == 3:
        args = run("--sensor-idxs 0 --epoch 1".split())
        show_confusion_matrix(args.trainer.confusion_matrix)

if __name__ == "__main__":
    test_code(3)
