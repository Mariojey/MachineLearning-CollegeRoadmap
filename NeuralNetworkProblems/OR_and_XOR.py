import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_OR = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y_OR = np.array([0,1,1,1])

X_XOR = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

Y_XOR = np.array([0,1,1,0])

model_OR = LogisticRegression()
model_XOR = LogisticRegression()

model_OR.fit(X_OR, Y_OR)
model_XOR.fit(X_XOR, Y_XOR)

pred_OR = model_OR.predict(X_OR)
pred_XOR = model_XOR.predict(X_XOR)

print("For OR problem")
print("Prediction from logistic regression:", pred_OR)
print("Accuracy:", accuracy_score(Y_OR, pred_OR))

print("\nFor XOR problem")
print("Prediction from logistic regression:", pred_XOR)
print("Accuracy:", accuracy_score(Y_XOR, pred_XOR))