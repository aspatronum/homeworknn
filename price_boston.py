#波士顿房价预测
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import logging

def main():
    # 设置日志记录
    logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler("training.log"),
                                logging.StreamHandler()
                            ])
    # 1.导入数据
    data=pd.read_excel("BostonHousingData.xlsx")
    # 2.数据预处理
    data=data.dropna() # 删除缺失值
    data=data.drop_duplicates() # 删除重复值
    # 前12列作为特征，MEDV作为目标变量
    X=data.iloc[:,:12].values
    y=data['MEDV'].values
    # 3.数据划分：前450条训练，后50条测试
    X_train=X[:450]
    y_train=y[:450]
    X_test=X[-50:]
    y_test=y[-50:]
    # 4.数据标准化和PCA降维
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    pca=PCA(n_components=0.9) # 保留90%的方差
    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)
    logging.info(f"原始特征数: {X.shape[1]}")
    logging.info(f"经过PCA降维后的特征数: {X_train.shape[1]}")
    # 5.数据转换为numpy数组
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    # 数据转换为PyTorch的tensor，并调整目标维度为二维张量
    X_train_tensor=torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor=torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor=torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor=torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    # 6.创建训练数据集与数据加载器
    train_dataset=TensorDataset(X_train_tensor, y_train_tensor)
    train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True)
    # 7.定义神经网络模型
    class RegressionNN(nn.Module):
        def __init__(self, input_dim):
            super(RegressionNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.model(x)
    input_dim = X_train_tensor.shape[1]
    model = RegressionNN(input_dim)
    # 8.定义损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 9.训练模型
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'boston_model.pth')
    logging.info("模型已保存为 boston_model.pth")
    # 加载模型
    model.load_state_dict(torch.load('boston_model.pth'))    
    
    # 10.在测试集上评估模型
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        logging.info(f'Test Loss: {test_loss.item():.4f}')

if __name__ == "__main__":
    main()