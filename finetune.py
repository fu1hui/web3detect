import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

# 鍔犺浇棰勫厛璁粌鐨? BERT 妯″瀷
tokenizer = BertTokenizer.from_pretrained(r'.\bert-base-uncased')

# 瀹氫箟鏁版嵁闆?
class EVMTransactionDataset(Dataset):
    def __init__(self, data_list, labels):
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_map = self.data_list[index]
        label = self.labels[index]

        # 灏? dataMap 杞崲涓? BERT 杈撳叆鏍煎紡锛屽苟娣诲姞 padding
        inputs = tokenizer.encode_plus(
            data_map,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',  # 娣诲姞 padding
            truncation=True,  # 濡傛灉瓒呰繃 max_length 鍒欐埅鏂?
            return_attention_mask=True,
            return_tensors='pt'
        )

        # 鐢变簬杩斿洖鐨勬槸寮犻噺锛屾垜浠渶瑕佸皢缁村害鍘嬬缉锛堜緥濡? [1, 512] -> [512]锛?
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }, label


# 浠庢枃浠跺す鍔犺浇鏁版嵁
def load_data_from_folder(folder_path, label_value):
    data_list = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = f.read().strip()
                data_list.append(data)
                labels.append(label_value)
    return data_list, labels


# 鍔犺浇鏀诲嚮鍜屾櫘閫氫氦鏄撴暟鎹?
attack_data_path = './processed_attack_data'
normal_data_path = './processed_normal_data'

attack_data, attack_labels = load_data_from_folder(attack_data_path, 1)  # 鏀诲嚮浜ゆ槗鏍囩璁句负1
normal_data, normal_labels = load_data_from_folder(normal_data_path, 0)  # 鏅€氫氦鏄撴爣绛捐涓?0

# 鍚堝苟鏁版嵁
all_data = attack_data + normal_data
all_labels = attack_labels + normal_labels

# 鍒掑垎璁粌闆嗗拰娴嬭瘯闆嗭紝80%浣滀负璁粌闆嗭紝20%浣滀负娴嬭瘯闆?
train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=0.2,
                                                                    random_state=42, stratify=all_labels)

# 鍒涘缓鏁版嵁闆嗗拰鏁版嵁鍔犺浇鍣?
dataset_train = EVMTransactionDataset(train_data, train_labels)
dataset_test = EVMTransactionDataset(test_data, test_labels)

dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=32, shuffle=False)


class FineTuneModel(nn.Module):
    def __init__(self):
        super(FineTuneModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.6)  # 澧炲姞 Dropout 鐨勬瘮渚?
        self.fc1 = nn.Linear(768, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)  # 娣诲姞 Batch Normalization
        self.fc2 = nn.Linear(128, 2)
        self.batch_norm2 = nn.BatchNorm1d(2)  # 娣诲姞 Batch Normalization
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs = self.bert(x['input_ids'].to(device), attention_mask=x['attention_mask'].to(device))
        x = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)  # Batch Normalization
        x = self.relu(x)
        x = self.fc2(x)
        return x

def save_embeddings(embeddings, labels, filename):
    np.savez(filename, embeddings=embeddings, labels=labels)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tune_model = FineTuneModel().to(device)

# 璁＄畻绫诲埆鏉冮噸
class_count = [len(normal_labels), len(attack_labels)]  # [6000, 600]
class_weight = [sum(class_count) / c for c in class_count]  # 鏍规嵁绫诲埆鏁伴噺璁＄畻鏉冮噸
weight_tensor = torch.FloatTensor(class_weight).to(device)

# 鏇存柊鎹熷け鍑芥暟锛屾坊鍔犳潈閲?
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
# 鍒涘缓浼樺寲鍣?
optimizer = optim.Adam(fine_tune_model.parameters(), lr=2e-5, weight_decay=1e-4)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

scaler = torch.amp.GradScaler('cuda')

# 璁板綍鎹熷け鍜屽噯纭巼
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# 鍒濆鍖? embedding 鍜屾爣绛剧殑淇濆瓨鍒楄〃
train_embeddings = []
train_labels_list = []

test_embeddings = []
test_labels_list = []

# 璁粌寰皟妯″瀷
for epoch in range(20):
    fine_tune_model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch in dataloader_train:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = fine_tune_model(inputs)
        loss = criterion(outputs, labels)
        scaler.scale(loss).backward()  # 姊害缂╂斁
        scaler.step(optimizer)  # 鏇存柊鍙傛暟
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    # 璁板綍璁粌鎹熷け鍜屽噯纭巼
    train_loss = running_loss / len(dataloader_train)
    train_accuracy = correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # 璇勪及寰皟妯″瀷
    fine_tune_model.eval()
    test_loss = 0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for batch in dataloader_test:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = fine_tune_model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    # 璁板綍娴嬭瘯闆嗙殑鎹熷け鍜屽噯纭巼
    test_loss = test_loss / len(dataloader_test)
    test_accuracy = correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # 杈撳嚭璁粌杩囩▼
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 瀛︿範鐜囪皟搴﹀櫒
    # scheduler.step(test_loss)

# 淇濆瓨鏈€缁堣缁冮泦 embedding
fine_tune_model.eval()
with torch.no_grad():
    for batch in dataloader_train:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 鎻愬彇璁粌闆? embedding
        embeddings = fine_tune_model.bert(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state[:, 0, :]
        train_embeddings.append(embeddings.cpu().detach().numpy())
        train_labels_list.append(labels.cpu().detach().numpy())

train_embeddings = np.vstack(train_embeddings)
train_labels_list = np.concatenate(train_labels_list)
# save_embeddings(train_embeddings, train_labels_list, 'train_embeddings.npz')

# 淇濆瓨鏈€缁堟祴璇曢泦 embedding
with torch.no_grad():
    for batch in dataloader_test:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}

        # 鎻愬彇娴嬭瘯闆? embedding
        embeddings = fine_tune_model.bert(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state[:, 0, :]
        test_embeddings.append(embeddings.cpu().detach().numpy())
        test_labels_list.append(labels.cpu().detach().numpy())

test_embeddings = np.vstack(test_embeddings)
test_labels_list = np.concatenate(test_labels_list)
# save_embeddings(test_embeddings, test_labels_list, 'test_embeddings.npz')

# 淇濆瓨妯″瀷
model_save_path = 'fine_tuned_model.pth'
torch.save(fine_tune_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 璁＄畻骞惰緭鍑? F1 鍒嗘暟绛夋寚鏍?
from sklearn.metrics import classification_report

# 璁＄畻璁粌闆嗙殑娣锋穯鐭╅樀
train_all_labels = []
train_all_preds = []

with torch.no_grad():
    for batch in dataloader_train:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        outputs = fine_tune_model(inputs)
        _, predicted = torch.max(outputs, 1)

        train_all_labels.extend(labels.cpu().numpy())
        train_all_preds.extend(predicted.cpu().numpy())

# 鎵撳嵃璁粌闆嗗垎绫绘姤鍛婂拰娣锋穯鐭╅樀
train_conf_matrix = confusion_matrix(train_all_labels, train_all_preds)
print("Train Confusion Matrix:")
print(train_conf_matrix)
print("\nTrain Classification Report:")
print(classification_report(train_all_labels, train_all_preds, target_names=['Normal', 'Attack']))

# 璁＄畻骞惰緭鍑烘祴璇曢泦鐨? F1 鍒嗘暟绛夋寚鏍?
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in dataloader_test:
        inputs, labels = batch
        inputs = {key: val.to(device) for key, val in inputs.items()}
        labels = labels.to(device)

        outputs = fine_tune_model(inputs)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# 鎵撳嵃娴嬭瘯闆嗗垎绫绘姤鍛婂拰娣锋穯鐭╅樀
test_conf_matrix = confusion_matrix(all_labels, all_preds)
print("Test Confusion Matrix:")
print(test_conf_matrix)
print("\nTest Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Normal', 'Attack']))


# 缁樺埗璁粌鎹熷け鍜屾祴璇曟崯澶?
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='orange')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# 缁樺埗璁粌鍑嗙‘鐜囧拰娴嬭瘯鍑嗙‘鐜?
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='green')
plt.plot(test_accuracies, label='Test Accuracy', color='red')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# 淇濆瓨鍥惧儚
plt.tight_layout()
plt.savefig('training_results.png')
plt.show()