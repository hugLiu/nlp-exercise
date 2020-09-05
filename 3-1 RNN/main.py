import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# 定义超参数
TIME_STEP = 10
INPUT_SIZE = 1
learning_rate = 0.001

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # r_out.shape:seq_len,batch,hidden_size*num_direction(1,10,32)
        r_out, h_state = self.rnn(x, h_state)
        out = self.out(r_out).squeeze()
        return out, h_state

rnn = RNN()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

h_state = None

plt.figure(1, figsize=(12, 5))
plt.ion() # 开启动态交互

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32, endpoint=False)
    x_np = np.sin(steps) # x_np.shape: 10
    y_np = np.cos(steps) # y_np.shape: 10

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]) # x.shape: 1,10,1
    y = torch.from_numpy(y_np) # y.shape: 10

    prediction, h_state = rnn(x, h_state)
    h_state = h_state.data

    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(.05)

plt.ioff()
plt.show()