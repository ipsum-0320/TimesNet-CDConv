import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, output_step):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.output_step = output_step
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size * self.output_step)

    def forward(self, input_seq):
        _, (h_n, _) = self.lstm(input_seq)

        # 使用最后一个隐藏状态生成未来 30 个时间步的预测
        output = self.linear(h_n[-1])  # h_n[-1] 形状为 [batch_size, hidden_size]

        # 将输出形状调整为 [batch_size, output_steps, output_size]
        output = output.view(-1, self.output_step, self.output_size)
        return output
