import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertTokenizer
from finetune import FineTuneModel

tokenizer = BertTokenizer.from_pretrained(r'.\bert-base-uncased')


# 加载模型
bert_model = FineTuneModel()
bert_model.load_state_dict(torch.load('./fine_tuned_model.pth'))
bert_model.eval()


# load data
def load_data(txt_folder_path, json_folder_path, label: int):
    txt_data_list = []
    json_data_list = []
    labels_list = []
    txt_files = {f.split('.')[0]: f for f in os.listdir(txt_folder_path) if f.endswith('.txt')}
    json_files = {f.split('.')[0]: f for f in os.listdir(json_folder_path) if f.endswith('.json')}

    for file_name in txt_files.keys():
        if file_name in json_files:
            with open(os.path.join(txt_folder_path, file_name), 'r') as txt_file:
                txt_data = txt_file.read().strip()

            with open(os.path.join(json_folder_path, json_files[file_name]), 'r') as json_file:
                json_data = json.load(json_file)
        txt_data_list.append(txt_data)
        json_data_list.append(json_data)
        labels_list.append(label)

    return txt_data, json_data_list, labels_list


# data path
attack_data_txt_path = './processed_attack_data'
attack_data_json_path = './attack_json'
normal_data_txt_path = './processed_normal_data'
normal_data_json_path = './normal_json'

# 加载数据
attack_embadding, attack_7f, attack_labels = load_data(attack_data_txt_path, attack_data_json_path, 1)
normal_embadding, normal_7f, normal_labels = load_data(normal_data_txt_path, normal_data_json_path, 0)
X_embedding = attack_embadding + normal_embadding
X_statistical = attack_7f + normal_7f
y = attack_labels + normal_labels


# 数据集定义
class EVMTransactionDataset(Dataset):
    def __init__(self, X_statistical, X_embedding, y, bert_model, tokenizer, device):
        self.X_statistical = torch.tensor(X_statistical, dtype=torch.float32)
        self.X_embedding = X_embedding
        self.y = torch.tensor(y, dtype=torch.float32)
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.X_statistical)

    def __getitem__(self, index):

        encoding = self.tokenizer.encode_plus(
            self.X_embedding[index],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0).to(self.device)
        attention_mask = encoding['attention_mask'].squeeze(0).to(self.device)

        # 使用微调过的 BERT 模型获取嵌入
        with torch.no_grad():
            bert_embedding = self.bert_model.embeddings(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        bert_embedding = torch.tensor(bert_embedding, dtype=torch.long).to(self.device)

        return self.X_statistical[index].to(self.device), bert_embedding, self.y[index].to(self.device)


# 分割数据集
train_size = int(0.8 * len(X_statistical))
X_statistical_train, X_statistical_test = X_statistical[:train_size], X_statistical[train_size:]
X_embedding_train, X_embedding_test = X_embedding[:train_size], X_embedding[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 创建数据集与加载器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_train = EVMTransactionDataset(X_statistical_train, X_embedding_train, y_train, bert_model, tokenizer, device)
dataset_test = EVMTransactionDataset(X_statistical_test, X_embedding_test, y_test, bert_model, tokenizer, device)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)

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


# 模型初始化
model = ImprovedNet().to(device)
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
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader_train):.4f}')

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
