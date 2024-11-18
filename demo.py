import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 定义输入特征
statistical_features = 6  # 统计特征维度
embedding_vocab_size = 10000  # 嵌入特征词汇表大小
embedding_dim = 128  # 嵌入特征维度

# 定义模型结构
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        # 统计特征部分
        self.fc1 = nn.Linear(statistical_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.statistical_output = nn.Linear(32, 16)
        self.dropout = nn.Dropout(0.3)  # 防止过拟合

        # 嵌入特征部分
        self.embedding = nn.Embedding(embedding_vocab_size, embedding_dim)
        self.embedding_fc = nn.Linear(embedding_dim, 16)

        # 特征融合
        self.fusion_fc = nn.Linear(16 + 16, 32)

        # 输出层
        self.output_fc = nn.Linear(32, 1)

    def forward(self, x_statistical, x_embedding):
        # 统计特征处理
        x_statistical = torch.relu(self.fc1(x_statistical))
        x_statistical = self.dropout(torch.relu(self.fc2(x_statistical)))
        x_statistical = torch.relu(self.statistical_output(x_statistical))

        # 嵌入特征处理
        x_embedding = self.embedding(x_embedding).squeeze(1)  # 形状调整为 (batch_size, embedding_dim)
        x_embedding = torch.relu(self.embedding_fc(x_embedding))

        # 特征融合
        x = torch.cat((x_statistical, x_embedding), dim=1)  # 拼接特征
        x = torch.relu(self.fusion_fc(x))  # 融合后的全连接层

        # 输出层
        return torch.sigmoid(self.output_fc(x))  # 二分类概率输出

# 数据集定义
class EVMTransactionDataset(Dataset):
    def __init__(self, X_statistical, X_embedding, y):
        self.X_statistical = torch.tensor(X_statistical, dtype=torch.float32)
        self.X_embedding = torch.tensor(X_embedding, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_statistical)

    def __getitem__(self, index):
        return self.X_statistical[index], self.X_embedding[index], self.y[index]

# 数据生成与准备
np.random.seed(0)
X_statistical = np.random.rand(1000, statistical_features)  # 统计特征数据
X_embedding = np.random.randint(0, embedding_vocab_size, size=(1000, 1))  # 嵌入特征数据
y = np.random.randint(0, 2, size=(1000,))  # 标签数据

# 数据集分割
train_size = int(0.8 * len(X_statistical))
X_statistical_train, X_statistical_test = X_statistical[:train_size], X_statistical[train_size:]
X_embedding_train, X_embedding_test = X_embedding[:train_size], X_embedding[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 创建数据集与加载器
dataset_train = EVMTransactionDataset(X_statistical_train, X_embedding_train, y_train)
dataset_test = EVMTransactionDataset(X_statistical_test, X_embedding_test, y_test)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

# 模型初始化
model = ImprovedNet()
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):  # 训练10个epoch
    model.train()
    epoch_loss = 0
    for X_statistical, X_embedding, y in dataloader_train:
        optimizer.zero_grad()
        outputs = model(X_statistical, X_embedding)
        loss = criterion(outputs.squeeze(), y)  # BCELoss需要outputs和y形状一致
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader_train):.4f}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_statistical, X_embedding, y in dataloader_test:
        outputs = model(X_statistical, X_embedding).squeeze()
        predictions = (outputs > 0.5).float()  # 阈值为0.5
        correct += (predictions == y).sum().item()
        total += y.size(0)

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
