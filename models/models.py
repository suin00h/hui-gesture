import torch

from torch import nn

class DeepConvLSTM(nn.Module):
    def __init__(
        self, 
        sensor_channels,
        input_length, 
        kernel_size=5, 
        stride=1, 
        lstm_hidden_size=128, 
        lstm_layers=2, 
        num_classes=26
    ):
        super().__init__()
        
        self.sensor_channels = sensor_channels
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv_channels = [sensor_channels, 64, 64, 64, 64]
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        
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
            nn.Conv1d(in_channels, out_channels, 
                      self.kernel_size, self.stride),
            nn.ReLU()
        )
    
    def compute_conv_output_size(self):
        size = self.input_length
        for _ in range(len(self.conv_channels) - 1):
            size = (size - self.kernel_size) // self.stride + 1
        return size

    def get_hidden_state(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(2, batch_size, 128).zero_(),
                  weight.new(2, batch_size, 128).zero_())
        
        return hidden
    
    #def forward(self, x, h):
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        for conv in self.conv_list:
            x = conv(x)
        
        x = torch.transpose(x, 1, 2)
        #x, h = self.lstm(x, h)
        x, _     = self.lstm(x)
        
        x = x.contiguous().view(-1, self.fc_input_size)
        x = self.fc(x)
        
        # return x, h
        return x

if __name__ == "__main__":
    net = DeepConvLSTM(3, 400, kernel_size=10, stride=2)
    h = net.get_hidden_state(100)
    x = torch.randn((100, 400, 3))
    x_, h_ = net(x, h)
    print(x_.shape)
