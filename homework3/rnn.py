import urllib.request
import tarfile
import os
import re
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 下载 IMDB 数据集压缩包
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"
if not os.path.exists(filename):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")
else:
    print("Dataset archive already exists.")

# 解压缩 .tar.gz 文件
if not os.path.exists("aclImdb"):
    print("Extracting dataset files...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall()  # 解压出 aclImdb/ 目录
    print("Extraction complete.")
else:
    print("Dataset directory already exists.")

# 列出训练集和测试集目录内容
train_dir = "aclImdb/train"
test_dir = "aclImdb/test"
print("Train subfolders:", os.listdir(train_dir))
print("Test subfolders:", os.listdir(test_dir))
print("Number of training pos files:", len(os.listdir(os.path.join(train_dir, "pos"))))
print("Number of training neg files:", len(os.listdir(os.path.join(train_dir, "neg"))))
print("Number of test pos files:", len(os.listdir(os.path.join(test_dir, "pos"))))
print("Number of test neg files:", len(os.listdir(os.path.join(test_dir, "neg"))))


def clean_text(text):
    """清洗文本：去除HTML标签和标点，并转换为小写。"""
    # 去掉 HTML 标签
    text = re.sub(r"<.*?>", "", text)
    # 去掉标点符号（只保留字母、数字和空格）
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # 转换为小写
    text = text.lower()
    return text

# 测试清洗函数
sample_text = "I <br /> loved this Movie!!! It's great, though a bit long."
print("原始文本:", sample_text)
print("清洗后:", clean_text(sample_text))

# 准备容器
train_tokens = []   # 存储训练集每条评论的分词结果
train_labels = []   # 存储训练集每条评论的标签 (1或0)
test_tokens = []
test_labels = []

# 处理训练集
for label in ["pos", "neg"]:
    folder = os.path.join(train_dir, label)
    for filename in os.listdir(folder):
        # 读取评论文本
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
        # 清洗 + 分词
        text = clean_text(text)
        tokens = text.split()  # 按空格分词
        # 保存结果
        train_tokens.append(tokens)
        train_labels.append(1 if label == "pos" else 0)

# 处理测试集
for label in ["pos", "neg"]:
    folder = os.path.join(test_dir, label)
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
        text = clean_text(text)
        tokens = text.split()
        test_tokens.append(tokens)
        test_labels.append(1 if label == "pos" else 0)

# 打乱训练集数据顺序（测试集无需打乱）
# 以防止所有正面评论和负面评论分别集中

train_data = list(zip(train_tokens, train_labels))
random.shuffle(train_data)
train_tokens, train_labels = zip(*train_data)  # 打乱后拆分
train_tokens = list(train_tokens)
train_labels = list(train_labels)

# 简单检查处理结果
print("训练集评论数:", len(train_tokens))
print("测试集评论数:", len(test_tokens))
print("第1条训练评论分词结果:", train_tokens[0][:20], "...")  # 打印前20个词作为示例
print("对应标签:", train_labels[0])

# 统计训练集词频
counter = Counter()
for tokens in train_tokens:
    counter.update(tokens)
print("训练集中的不同单词数（未经限制）:", len(counter))

# 限定词汇表大小
max_vocab_size = 10000  # 我们只保留最常见的10000个单词
most_common = counter.most_common(max_vocab_size)
print(f"词频最高的5个单词: {most_common[:5]}")

# 建立 单词->索引 映射
word2idx = {"<PAD>": 0, "<UNK>": 1}
for idx, (word, freq) in enumerate(most_common, start=2):
    word2idx[word] = idx
vocab_size = len(word2idx)
print("词汇表大小:", vocab_size)

# 将训练集和测试集的 token 序列转换为索引序列
train_sequences = []
for tokens in train_tokens:
    seq = [word2idx.get(word, 1) for word in tokens]  # 不在词表的词用<UNK>(索引1)替代
    train_sequences.append(seq)
test_sequences = []
for tokens in test_tokens:
    seq = [word2idx.get(word, 1) for word in tokens]
    test_sequences.append(seq)

# 设置最大序列长度并对序列进行填充/截断
MAX_LEN = 500  # 最大序列长度
def pad_or_truncate(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]    # 截断到最大长度
    else:
        return seq + [0] * (max_len - len(seq))  # 不足则填充0（<PAD>）

X_train_seq = [pad_or_truncate(seq, MAX_LEN) for seq in train_sequences]
X_test_seq  = [pad_or_truncate(seq, MAX_LEN) for seq in test_sequences]

# 转换为numpy数组以便构建张量
X_train_seq = np.array(X_train_seq)
X_test_seq = np.array(X_test_seq)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print("训练序列数组形状:", X_train_seq.shape)
print("测试序列数组形状:", X_test_seq.shape)
print("第一条训练序列 (索引形式):", X_train_seq[0][:20])
print("对应标签:", y_train[0])

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                             batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        # 根据是否双向决定全连接层输入维度
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, 1)
        else:
            self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: [batch_size, seq_len] 单词索引序列
        emb = self.embedding(x)               # emb: [batch_size, seq_len, embed_dim]
        lstm_out, (h_n, c_n) = self.lstm(emb) # h_n: [num_layers*num_directions, batch_size, hidden_dim]
        if self.bidirectional:
            # 双向LSTM: 前向最后一层隐状态在 h_n[-2], 后向最后一层隐状态在 h_n[-1]
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)  # 拼接前向和后向隐状态 => [batch_size, hidden_dim*2]
        else:
            # 单向LSTM: 最后一层隐状态即 h_n[-1]
            out = h_n[-1]  # [batch_size, hidden_dim]
        out = self.fc(out)           # 全连接层: [batch_size, 1]
        return out.squeeze(1)        # 输出展平成 [batch_size] 一维向量

# 定义超参数
vocab_size = len(word2idx)
embed_dim = 100
hidden_dim = 128
bidirectional = False  # 若想使用双向LSTM可设置为 True

# 实例化模型
model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, bidirectional=bidirectional)
print(model)

# GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 20
train_losses = []
test_accuracies = []

for epoch in range(1, num_epochs+1):
    model.train()  # 切换模型到训练模式
    total_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(X_batch)
        # 计算损失（注意：BCEWithLogitsLoss期望target为float）
        loss = loss_fn(outputs, y_batch)
        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)  # 累计损失总和（按样本加权）
    # 计算平均损失
    avg_loss = total_loss / len(train_loader.dataset)
    train_losses.append(avg_loss)
    
    # 在测试集上评估
    model.eval()  # 切换模型到评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            # 使用 sigmoid 将输出映射到 [0,1] 概率，然后以0.5为阈值判断正负面
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()  # 概率>=0.5则预测为1，否则为0
            # 统计预测正确的数量
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    acc = correct / total
    test_accuracies.append(acc)
    
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Test Accuracy = {acc:.4f}")

epochs = range(1, num_epochs+1)
plt.figure(figsize=(8,4))
plt.plot(epochs, train_losses, '-o', label='Training Loss')
plt.plot(epochs, test_accuracies, '-o', label='Test Accuracy')
plt.title('Training Loss and Test Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

model.eval()  # 确保模型在评估模式
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
test_accuracy = correct / total
print(f"在测试集上的准确率: {test_accuracy:.4f}")
