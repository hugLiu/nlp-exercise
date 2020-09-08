# 1. 导库
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# 1.1 配置，运行环境
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 数据处理
letter = [c for c in 'SE?abcdefghijklmnopqrstuvwxyz']
letter2idx = {l:i for i, l in enumerate(letter)}
seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

# 2.1 超参数
letter_size = len(list(set(letter)))
seq_len = max(max(len(i), len(j)) for i, j in seq_data)
hidden_size = 128
batch_size = 2

# 2.2 生成input,target，构造dataset, dataloader
def get_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + '?' * (seq_len - len(seq[i]))
        enc_input = [letter2idx[j] for j in (seq[0] + 'E')]
        dec_input = [letter2idx[j] for j in ('S' + seq[1])]
        dec_output = [letter2idx[j] for j in (seq[1] + 'E')]

        enc_inputs.append(np.eye(letter_size)[enc_input])
        dec_inputs.append(np.eye(letter_size)[dec_input])
        dec_outputs.append(dec_output)
    return enc_inputs, dec_inputs, dec_outputs

enc_inputs, dec_inputs, dec_outputs = get_data()
enc_inputs, dec_inputs, dec_outputs = torch.Tensor(enc_inputs), torch.Tensor(dec_inputs), torch.LongTensor(dec_outputs)

# 2.3 有3个数据，dataset需自定义
class TransDataset(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return len(enc_inputs)

    def __getitem__(self, index):
        return enc_inputs[index], dec_inputs[index], dec_outputs[index]

dataset = TransDataset(enc_inputs, dec_inputs, dec_outputs)
dataloader = DataLoader(dataset, batch_size, True)

# 3. 模型
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.RNN(letter_size, hidden_size)
        self.decoder = nn.RNN(letter_size, hidden_size)
        self.fc = nn.Linear(hidden_size, letter_size)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)
        dec_input = dec_input.transpose(0, 1)
        # enc_input [seq_len, batch, input_size] 6,2,29
        # enc_hidden [num_layers * num_directions, batch, hidden_size] 1,2,128
        # h_t [num_layers * num_directions, batch, hidden_size] 1,2,128
        _, h_t = self.encoder(enc_input, enc_hidden)
        # dec_input [seq_len, batch, input_size]
        # out [seq_len, batch, num_directions * hidden_size] 6,2,128
        out, _ = self.decoder(dec_input, h_t)

        out = self.fc(out)
        return out

# 3.1 优化器，损失定义
model = Seq2Seq().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 4. 训练
for epoch in range(1000):
    for i, (enc_input_d, dec_input_d, dec_output_d) in enumerate(dataloader):
        # h_0 : [num_layers * num_directions, batch, hidden_size]
        h_0 = torch.zeros(1, batch_size, hidden_size).to(device)
        # enc_input_d [seq_len, batch, input_size]
        # dec_output_d [batch, len(seq_data)]
        enc_input_d, dec_input_d, dec_output_d = enc_input_d.to(device), dec_input_d.to(device), dec_output_d.to(device)
        # pred [seq_len, batch, letter_size]
        pred = model(enc_input_d, h_0, dec_input_d)

        # pred [batch, seq_len, letter_size]
        pred = pred.transpose(0, 1)
        loss = .0
        # 对每一项输入输出求损失
        for j in range(len(dec_output_d)):
            loss = loss + criterion(pred[j], dec_output_d[j])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if((epoch + 1) % 200 == 0) and i == 0:
            print((epoch + 1), loss)

# 5. 测试
for seq in seq_data:
    enc_test = seq[0] + '?' * (seq_len - len(seq[0]))
    dec_test = seq[1] + '?' * (seq_len - len(seq[1]))

    enc_input_test = [letter2idx[j] for j in (enc_test + 'E')]
    dec_input_test = [letter2idx[j] for j in ('S' + dec_test)]

    enc_inputs_test, dec_inputs_test = [], []
    enc_inputs_test.append(np.eye(letter_size)[enc_input_test])
    dec_inputs_test.append(np.eye(letter_size)[dec_input_test])

    hidden_test = torch.zeros(1, 1, hidden_size).to(device)

    out_test = model(torch.Tensor(enc_inputs_test).to(device), hidden_test, torch.Tensor(dec_inputs_test).to(device))

    pred_test = out_test.data.argmax(2)
    decoded = [letter[i] for i in pred_test]
    decoded_word = ''.join(decoded[:decoded.index('E')])

    print(enc_test, '->', decoded_word)