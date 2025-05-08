import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 데이터 로드 및 7000개 사용
df = pd.read_csv("train.csv")
df = df.iloc[:7000, :]

X = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype('uint8')
y = df.iloc[:, 0].values

X_tensor = torch.tensor(X).float().unsqueeze(1) / 255.0
X_tensor = F.interpolate(X_tensor, size=(224, 224), mode='bilinear')
X_tensor = X_tensor.repeat(1, 3, 1, 1)
y_tensor = torch.tensor(y).long()

full_dataset = TensorDataset(X_tensor, y_tensor)

# 클래스별 분할
indices_0_3 = [i for i in range(len(y)) if 0 <= y[i] <= 3]
indices_4_6 = [i for i in range(len(y)) if 4 <= y[i] <= 6]
indices_7_9 = [i for i in range(len(y)) if 7 <= y[i] <= 9]
indices_all = list(range(len(y)))

# 데이터 로더
train_loader_0_3 = DataLoader(Subset(full_dataset, indices_0_3), batch_size=32, shuffle=True)
train_loader_4_6 = DataLoader(Subset(full_dataset, indices_4_6), batch_size=32, shuffle=True)
train_loader_7_9 = DataLoader(Subset(full_dataset, indices_7_9), batch_size=32, shuffle=True)
train_loader_all  = DataLoader(Subset(full_dataset, indices_all), batch_size=32, shuffle=True)

# 테스트 데이터 설정
test_indices = np.random.choice(len(y), size=int(0.2 * len(y)), replace=False)
test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=32, shuffle=False)

# 모델 정의
def get_model():
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    return model.to(device)

# 학습 함수 (에포크 10개로 변경)
def local_train(model, loader, epochs=10):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# FedAvg 평균
def average_weights(models):
    avg_model = get_model()
    state_dicts = [model.state_dict() for model in models]
    avg_dict = {}
    for key in state_dicts[0]:
        avg_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    avg_model.load_state_dict(avg_dict)
    return avg_model

# 평가 함수
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# 연합학습 (3 클라이언트)
print("\n--- Local Training (0~3, 4~6, 7~9) ---")
client1 = get_model()
client2 = get_model()
client3 = get_model()
local_train(client1, train_loader_0_3, epochs=10)
local_train(client2, train_loader_4_6, epochs=10)
local_train(client3, train_loader_7_9, epochs=10)

print("\n--- FedAvg Aggregation ---")
fedavg_model = average_weights([client1, client2, client3])
acc_fedavg = evaluate(fedavg_model, test_loader)
print(f"FedAvg Aggregated Model Accuracy: {acc_fedavg:.4f}")

# 전체 데이터 학습
print("\n--- Training Full Model (0~9) ---")
full_model = get_model()
local_train(full_model, train_loader_all, epochs=10)
acc_full = evaluate(full_model, test_loader)
print(f"Full Model Accuracy (Trained on 0~9): {acc_full:.4f}")
