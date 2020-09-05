# 1. 导入基础包，全局参数
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 语料及语料库大小, 词与索引关系
sentences = ['i like dog', 'jack hate coffee', 'i love milk', 'jack study natural language process',
             'word2vec conclude skip-gram and cbow model', 'jack like coffee', 'dog coffee milk']


word_list = ' '.join(sentences).split()
vocab = list(set(word_list))
vocab_size = len(vocab)

word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = {i:w for i, w in enumerate(vocab)}

# 3. 窗口，skip_gram, 输入输出
window = 2
batch_size = 8

# 生成skip_gram
skip_gram = []
for center_idx in range(len(word_list)):
    center = word2idx[word_list[center_idx]]
    for context_idx in (list(range(center_idx - window, center_idx))
                                 + list(range(center_idx + 1, center_idx + 1 + window))):
        if context_idx < 0 or context_idx > len(word_list) - 1:
            continue
        else:
            context = word2idx[word_list[context_idx]]
        skip_gram.append([center, context])

def get_data():
    input_data = []
    target_data = []
    for i in range(len(skip_gram)):
        input_data.append(np.eye(vocab_size)[skip_gram[i][0]])
        target_data.append(skip_gram[i][1])
    return input_data, target_data

input, target = get_data()
input, target = torch.Tensor(input), torch.LongTensor(target)

# 4. 形成训练所需的dataloader
dataset = TensorDataset(input, target)
dataloder = DataLoader(dataset, batch_size, True)

# 5. 模型实现，优化器，损失函数
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.embed_size = 2
        self.W = nn.Parameter(torch.randn(vocab_size, self.embed_size).type(torch.Tensor))
        self.V = nn.Parameter(torch.randn(self.embed_size, vocab_size).type(torch.Tensor))

    def forward(self, x):
        # x[batch_size, vocab_size] one_hot
        out = torch.mm(torch.mm(x, self.W), self.V)
        return out

model = Word2Vec().to(device)
criteriom = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. 训练
for epoch in range(2000):
    for i, (input_x, target_y) in enumerate(dataloder):
        input_x = input_x.to(device)
        target_y = target_y.to(device)
        pred = model(input_x)
        loss = criteriom(pred, target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0 and i == 0:
            print(epoch + 1, loss.item())

# 7. 通过图像展示向量之间的关系
for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

plt.show()