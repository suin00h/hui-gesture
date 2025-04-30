import torch

from torch import nn

class DeepConvLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args.in_channels # List[int]
        self.out_channel = 64
        self.kernel_size = args.kernel_size
        self.stride = args.stride
        self.concat_latent = args.concat_latent
        self.lstm_hidden_size = args.lstm_hidden_size
        self.lstm_layers = args.lstm_layers
        self.input_length = args.input_length
        self.num_class = args.num_class
        self.fc_input_size = self.get_fc_input_size()
        
        if not self.concat_latent:
            self.in_channels = [sum(self.in_channels)]
        
        self.conv_list = nn.ModuleList()
        for in_channel in self.in_channels:
            self.conv_list.append(
                ConvBlock(
                    self.kernel_size,
                    self.stride,
                    in_channel,
                    self.out_channel
                )
            )
        
        self.lstm = nn.LSTM(
            self.out_channel * len(self.in_channels),
            self.lstm_hidden_size,
            self.lstm_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, self.num_class),
            nn.Softmax(dim=1)
        )
    
    def get_fc_input_size(self):
        return self.lstm_hidden_size * self.compute_conv_output_size()
    
    def compute_conv_output_size(self):
        size = self.input_length
        for _ in range(4):
            size = (size - self.kernel_size) // self.stride + 1
        return size
    
    def forward(self, x_list):
        """
        x_list: List[Tensor(B, Length, C)]
        """
        if not self.concat_latent:
            x_list = [torch.cat(x_list, dim=2)]
        latent_list = []
        for i, conv_block in enumerate(self.conv_list):
            x = torch.transpose(x_list[i], 1, 2)
            x = conv_block(x)
            latent_list.append(x)
        x = torch.cat(latent_list, dim=1)
        
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        
        x = x.contiguous().view(-1, self.fc_input_size)
        x = self.fc(x)
        
        return x

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, stride, in_channel, out_channel):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv_block = self.get_conv_block(in_channel, out_channel)
    
    def get_conv_block(self, in_channel, out_channel):
        conv_block = [self.get_conv_layer(in_channel, out_channel)]
        for _ in range(3):
            conv_block.append(self.get_conv_layer(out_channel, out_channel))
        return nn.Sequential(*conv_block)
    
    def get_conv_layer(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, self.kernel_size, self.stride),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        
        return x
