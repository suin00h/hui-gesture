import torch
import torch.nn as nn
import numpy as np

from tqdm.auto import tqdm
from torch import optim
from data.dataset import Sign_Language_Dataset
from models.models import DeepConvLSTM

def train(dataloader, net, opt, crit, device):
    net.train()
    
    h_0, c_0 = net.get_hidden_state(100)
    hidden = (h_0.to(device), c_0.to(device))
    
    train_loss = []
    for batch in dataloader:
        input_imu = batch["imu_acc"].float().to(device)
        label = batch["label"].long().to(device)
        hidden = tuple([each.data for each in hidden])
        
        opt.zero_grad()
        
        # class_output, hidden = net(input_imu, hidden)
        class_output = net(input_imu)
        
        loss = crit(class_output, label)
        train_loss.append(loss.item())
        loss.backward()
        opt.step()
    
    return train_loss

def test():
    ...

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Sign_Language_Dataset()
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=100, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    net = DeepConvLSTM(3, 400, kernel_size=10, stride=2)
    opt = optim.Adam(net.parameters(), lr=0.02)
    crit = nn.CrossEntropyLoss()
    
    train_loss_list = []
    
    net.to(device)
    for e in tqdm(range(1)):
        train_loss = train(train_dataloader, net, opt, crit, device)
        train_loss_list.append(np.mean(train_loss))
        
        test()

if __name__ == "__main__":
    run()
