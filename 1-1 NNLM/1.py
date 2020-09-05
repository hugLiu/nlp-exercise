# 1. 导入所需库
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# 2. 数据处理， 前2个词预测第3个词
sentence1 = 'i like dog'
sentence2 = 'i hate coffee'
sentence3 = 'i love milk'
sentence4 = 'i am study'
sentence5 = 'i miss you'

# 2.1 获取词典， word2idx, idx2word
def get_word_list():
    word_list = []
    for i in range(1, 6):
        s = eval('sentence' + str(i))
        word_list.append(s)
    word_list = ' '.join(word_list).split()
    return word_list

vocab = list(set(get_word_list()))
word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = {i:w for i, w in enumerate(vocab)}

# 2.2 生成input, target数据，及训练需要的dataloader
#     input取前第n-1个词索引，target取第n个词索引
def get_data():
    input = []
    target = []
    for i in range(1, 6):
        s = eval('sentence' + str(i))
        input.append([word2idx[w] for w in s.split()[:-1]])
        target.append(word2idx[s.split()[-1]])
    return input, target

input, target = get_data()
input, target = torch.LongTensor(input), torch.LongTensor(target)

batch_size = 5 # 验证时，需要全部句子的输入，即为句子数
dataset = TensorDataset(input, target)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 2.3 定义超参数
V = len(vocab) # 字典的大小
m = 100 # 词向量维度，C表的大小[V,m]
n = 3 # 句子长度，前n-1=2
h = 20 # 隐藏层神经元个数
c = (n - 1) * m

# 3. 定义模型类
# model: y = b + Wx + Utanh(d + Hx)
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(V, m)
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(batch_size)))
        self.W = nn.Parameter(torch.FloatTensor(torch.randn(V, c)))
        self.U = nn.Parameter(torch.FloatTensor(torch.randn(V, h)))
        self.d = nn.Parameter(torch.FloatTensor(torch.randn(batch_size)))
        self.H = nn.Parameter(torch.FloatTensor(torch.randn(h, c)))

    def forward(self, x):
        x = self.C(x) # [batch_size, n-1, m]
        x = x.view(-1, c) # x[batchsize, c]
        x = x.transpose(0, 1) # x[c, batchsize]
        output = self.b + torch.mm(self.W, x) + torch.mm(self.U, torch.tanh(self.d + torch.mm(self.H, x)))
        return output # [V, batchsize]

model = NNLM()

# 4. 定义优化器，损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. 训练
for epoch in range(1000):
    for i, (input_x, target_y) in enumerate(dataloader):
        pred = model(input_x)
        pred = pred.transpose(0, 1) # CrossEntorpyLoss[N, C] N为batch_size, 故需要转置
        loss = criterion(pred, target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(epoch + 1, loss.item())

# 6. 验证
preds = model(input).argmax(0)
print([idx2word[idx.item()] for idx in preds])