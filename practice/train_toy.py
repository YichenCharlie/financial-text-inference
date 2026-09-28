import torch
from torch import nn

torch.manual_seed(42)

# 每行一条 sample，每条只有一个输入特征
X = torch.tensor([
    [-2.0],
    [-1.0],
    [1.0],
    [2.0],
])

# 每条 sample 的正确类别
y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

model = nn.Linear(1, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 保存 training 前的 weight（权重，一类可学习参数）
weight_before = model.weight.detach().clone()
print("X shape：", X.shape)
print("输出 shape：", model(X).shape)
print("初始输出分数：\n", model(X).detach())

for step in range(100):
    optimizer.zero_grad()

    logits = model(X)             # ① 让 model 根据 X 输出分数
    loss = criterion(logits, y)

    loss.backward()                     # ② 根据 loss 计算 gradient
    optimizer.step()                 # ③ 用 optimizer 更新参数

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

print("\nTraining 前的 weight：")
print(weight_before)

print("\nTraining 后的 weight：")
print(model.weight.detach())

with torch.no_grad():
    scores = model(X)
    predictions = scores.argmax(dim=1)

print("\n预测类别：", predictions.tolist())
print("正确类别：", y.tolist())


# 这些数值没有出现在刚才的 training data 中
X_new = torch.tensor([
    [-3.0],
    [-0.2],
    [0.2],
    [3.0],
])

with torch.no_grad():
    new_scores = model(X_new)
    new_predictions = new_scores.argmax(dim=1)

print("\n新输入：", X_new.squeeze(1).tolist())
print("新输入的分数：\n", new_scores)
print("新输入的预测：", new_predictions.tolist())
print("新输入的正确类别：", [0, 0, 1, 1])
print("Training 后的 bias：", model.bias.detach())