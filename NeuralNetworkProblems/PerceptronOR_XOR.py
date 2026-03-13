import torch
import torch.nn as nn
import torch.optim as optim

X_OR = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

Y_OR = torch.tensor([
    [0.],
    [1.],
    [1.],
    [1.]
])

X_XOR = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

Y_XOR = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

def train_model(X, y, epochs=5000):

    model = Perceptron()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        y_pred = model(X)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} Loss {loss.item():.4f}")

    return model

print("TRAIN ON")
model_OR = train_model(X_OR, Y_OR)

with torch.no_grad():
    pred = model_OR(X_OR)
    pred_class = (pred > 0.5).float()

print("Prediction for OR")
print(pred_class)

print("TRAIN XOR")
model_XOR = train_model(X_XOR, Y_XOR)

with torch.no_grad():
    pred = model_XOR(X_XOR)
    pred_class = (pred > 0.5).float()

print("Prediction for XOR:")
print(pred_class)