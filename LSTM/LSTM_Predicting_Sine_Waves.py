import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 生成正弦波数据
def generate_sin_wave(seq_length, num_sequences):
    x = np.linspace(0, 50, seq_length * num_sequences)
    y = np.sin(x)
    sequences = []
    for i in range(num_sequences):
        sequences.append(y[i * seq_length: (i + 1) * seq_length])
    return np.array(sequences)


# 准备数据
seq_length = 5
num_sequences = 1000
data = generate_sin_wave(seq_length, num_sequences)

# 生成输入和输出，并重塑为（seq_length, batch_size, input_size）
# x序列丢弃原始数据的最后一个
# y序列丢弃原始数据的第一个
x = data[:, :-1]
y = data[:, 1:]

# 转换为 PyTorch 的张量
x = torch.tensor(x, dtype=torch.float32).reshape(-1, seq_length - 1, 1)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, seq_length - 1, 1)


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 层前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 全连接层前向传播
        predictions = self.linear(out)

        return predictions


# 初始化模型、损失函数和优化器
model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 200  # 定义训练周期的数量，即模型将整个数据集上训练100次
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    outputs = model(x)  # 前向传播，计算模型的输出
    optimizer.zero_grad()  # 清零模型参数的梯度
    loss = criterion(outputs, y)  # 计算模型输出与真实标签之间的损失
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

    # 每20个周期打印一次当前的训练损失
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 可视化预测结果
model.eval()
with torch.no_grad():
    predictions = model(x)
    predictions = predictions.detach().numpy()

plt.plot(predictions.flatten(), label='Predicted')
plt.plot(y.flatten(), label='True')
plt.legend()
plt.savefig('LSTM_Predicting_Sine_Waves.png')  # 保存图像
