import torch

from torch import nn

class DeepConvLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.input_length = args.input_length
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.conv_channels = [sum(args.in_channels), 64, 64, 64, 64]
        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm_layers = args.lstm_layers
        self.num_classes = args.num_classes
        
        self.conv_list = nn.ModuleList([
            self.get_conv_layer(
                self.conv_channels[i],
                self.conv_channels[i+1],
            ) for i in range(len(self.conv_channels) - 1)
        ])
        self.lstm = nn.LSTM(
            self.conv_channels[-1],
            self.lstm_hidden_size,
            self.lstm_layers,
            batch_first=True
        )
        self.fc_input_size = self.compute_conv_output_size() \
            * self.lstm_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.num_classes),
            nn.Softmax(dim=1)
        )
    
    def get_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, self.kernel_size, self.stride),
            nn.ReLU()
        )
    
    def compute_conv_output_size(self):
        size = self.input_length
        for _ in range(len(self.conv_channels) - 1):
            size = (size - self.kernel_size) // self.stride + 1
        return size
    
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        for conv in self.conv_list:
            x = conv(x)
        
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        
        x = x.contiguous().view(-1, self.fc_input_size)
        x = self.fc(x)
        
        return x

class DeepConvLSTM_latent(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.input_length = args.input_length
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.in_channels = args.in_channels # [3, 3, 4, 8], [3, 4, 8]
        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm_layers = args.lstm_layers
        self.num_classes = args.num_classes
        
        self.conv_block_list = [self.get_conv_block(in_channel) for in_channel in self.in_channels]
        
        self.lstm = nn.LSTM(
            64 * len(self.in_channels),
            self.lstm_hidden_size,
            self.lstm_layers,
            batch_first=True
        )
        self.fc_input_size = self.compute_conv_output_size() \
            * self.lstm_hidden_size
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.num_classes),
            nn.Softmax(dim=1)
        )
    
    def get_conv_block(self, in_channel):
        conv_block = nn.ModuleList()
        conv_block.append(self.get_conv_layer(in_channel, 64))
        for _ in range(3):
            conv_block.append(self.get_conv_layer(64, 64))
        return conv_block
    
    def get_conv_layer(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, self.kernel_size, self.stride),
            nn.ReLU()
        )
    
    def compute_conv_output_size(self):
        size = self.input_length
        for _ in range(4):
            size = (size - self.kernel_size) // self.stride + 1
        return size
    
    def forward(self, x_list):
        latent_list = []
        for i, conv_block in enumerate(self.conv_block_list):
            x = torch.transpose(x_list[i], 1, 2)
            for conv in conv_block:
                x = conv(x)
            latent_list.append(x)
        
        x = torch.cat(latent_list, dim=1)
        
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        
        x = x.contiguous().view(-1, self.fc_input_size)
        x = self.fc(x)
        
        return x
