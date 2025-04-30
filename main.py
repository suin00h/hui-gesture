import os
import torch
import numpy as np

from torch import nn
from torch import optim
from sklearn import metrics
from datetime import datetime
from tqdm.auto import tqdm

from utils import *
from data.dataset import Sign_Language_Dataset
from models.models import DeepConvLSTM

def run(custom_arg=None):
    parser = get_parser()
    args = parser.parse_args(args=custom_arg)
    
    set_settings(args)
    set_device(args)
    set_dataloader(args, Sign_Language_Dataset(), [0.8, 0.1, 0.1])
    
    args.net = DeepConvLSTM(args)
    args.optimizer = optim.Adam(args.net.parameters(), lr=args.learning_rate)
    args.criterion = nn.CrossEntropyLoss()
    
    tracker = MetricTracker(num_class=args.num_class)
    logger = Logger(args)
    trainer = Trainer(args, tracker, logger)
    return
    trainer.run()
    
    return tracker

class Trainer():
    def __init__(self, args, tracker, logger):
        self.net = args.net
        self.opt = args.optimizer
        self.crit = args.criterion
        self.device = args.device
        
        self.epoch = args.epoch
        self.test_only = args.test_only
        self.in_sensors = args.in_sensors
        self.dataloaders = args.dataloaders
        
        self.tracker = tracker
        self.logger = logger
        
        logger(args.setting_description)
        logger.log_args(args)
    
    def run(self):
        self.network_to_device()
        
        if not self.test_only:
            for _ in tqdm(range(self.epoch)):
                self.run_epoch()
        
        self.step("test")
    
    def run_epoch(self):
        self.step("train")
        with torch.no_grad():
            self.step("val")
    
    def step(self, phase):
        self.tracker.set_step_metric(phase)
        is_train = phase == "train"
        if is_train:
            self.net.train()
        else:
            self.net.eval()
        
        for batch in self.dataloaders[phase]:
            sensor_input, label = self.process_batch(batch)
            
            if is_train:
                self.opt.zero_grad()
            net_output = self.net(sensor_input)
            loss = self.crit(net_output, label)
            
            if is_train:
                loss.backward()
                self.opt.step()
            
            self.tracker.update_step_metric(loss, net_output, label)
        
        self.tracker.step()
    
    def network_to_device(self):
        self.net.to(self.device)
    
    def process_batch(self, batch):
        sensor_input = self.get_sensor_list(batch)
        label = batch["label"].long().to(self.device)
        
        return sensor_input, label
    
    def get_sensor_list(self, batch):
        return [batch[sensor].float().to(self.device) for sensor in self.in_sensors]

class MetricTracker():
    def __init__(self, num_class):
        self.metric_list = ["loss", "accuracy", "f1score"]
        self.phase_list = ["train", "val", "test"]
        self.phase = ""
        
        self.metrics = {
            phase: self.get_metric_dict()
            for phase in self.phase_list
        }
        self.metric_step = {}
        self.confusion_matrix = np.zeros((num_class, num_class))
    
    def set_step_metric(self, phase):
        """
        Initialize metric lists for a single step
        """
        self.metric_step = self.get_metric_dict()
        self.phase = phase
    
    def update_step_metric(self, loss, output, label):
        """
        Called during a single batch processing
        Get models outputs and compute metrics
        """
        self.metric_step["loss"].append(loss.item())
        self.metric_step["accuracy"].append(self.get_accuracy(output, label))
        self.metric_step["f1score"].append(self.get_f1score(output, label))
        if self.phase == "test":
            self.confusion_matrix += self.get_confusion_matrix(output, label)
    
    def step(self):
        """
        Called after all batch iteration
        Accumulate metrics from batch
        """
        for key, value in self.metric_step.items():
            metric = np.mean(value).item()
            self.metrics[self.phase][key].append(metric)
    
    def get_accuracy(self, output, label):
        true_positive = torch.argmax(output, dim=1).squeeze() == label
        return true_positive.sum().item() / len(label)
    
    def get_f1score(self, output, label):
        return metrics.f1_score(
            torch.argmax(output, dim=1).cpu(),
            label.cpu(),
            average="macro"
        )
    
    def get_confusion_matrix(self, output, label):
        pred = torch.argmax(output, dim=1).squeeze().cpu()
        true = label.cpu()
        return metrics.confusion_matrix(true, pred, labels=np.arange(26))
    
    def get_metric_dict(self):
        return {metric: [] for metric in self.metric_list}

class Logger():
    def __init__(self, args):
        self.show_detail = False
        self.save_log = args.save_log
        
        self.payload = ""
        self.title = self.get_title(args)
        self.set_log_dir()
    
    def __call__(self, log_string):
        tqdm.write(log_string)
        self.payload += log_string + "\n"
    
    def write_payload(self):
        with open(self.log_dir + "/log.txt", "a") as log:
            log.write(self.payload)
        self.payload = ""
    
    def log_args(self, args):
        header = "Experiment Settings:\n" + "-" * 60
        args_str = "\n".join(f"{k:<20}: {v}" for k, v in vars(args).items())
        footer = "-" * 60
        full_str = f"{header}\n{args_str}\n{footer}"
        
        self(full_str)
        self.write_payload()
    
    def set_log_dir(self):
        self.log_dir = os.path.join("log", self.title)
        if self.save_log:
            os.makedirs(self.log_dir)
    
    def get_title(self, args):
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        sensors = ''.join(str(i) for i in args.sensor_idxs)
        concat = "latent-cat" if args.concat_latent else "input-cat"
        
        return (f"{timestamp}-{type(args.net).__name__}-b{args.batch_size}"
                f"-l{args.learning_rate:.0e}-sensor{sensors}-{concat}"
                f"-k{args.kernel_size}-s{args.stride}")

def test_code(code_idx):
    if code_idx == 1:
        args = run()    # run([]) or run("{custom arguments}".split()) for colab environment
        # print(args.metrics["loss"]["train"])
        # print(args.metrics["loss"]["test"])
        # print(args.metrics["f1score"]["test"])
        # print(args.metrics["accuracy"]["test"])
    elif code_idx == 2:
        p = torch.randint(0, 7, (7,))
        t = torch.randint(0, 7, (7,))
        cm = metrics.confusion_matrix(p, t, labels = np.arange(7))
        print(cm)
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(cm)
        plt.show()
    elif code_idx == 3:
        tracker = run("--sensor-idxs 0 1 3 --epoch 1 --save-log".split())
        #show_confusion_matrix(tracker.confusion_matrix)
    elif code_idx == 4:
        a = np.array([[1, 1], [2, 2]])
        b = np.array([[1, 1], [2, 2]])
        print(sum([a, b]))
    elif code_idx == 5:
        import os
        print(os.path.dirname(__file__))

if __name__ == "__main__":
    test_code(3)
