from sklearn.datasets import make_circles

# Make 1000 samples
n_samples = 1000

# Create circles, with some noise
X, y = make_circles(n_samples, noise = 0.03,
                    random_state = 42)

# View the first 5 X and y values.
print(f"First 5 X features:\n{X[:5]}")
print(f"\nFirst 5 y labels:\n{y[:5]}")

# Make dataframe of circle data.
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

circles.head(10)

# Check the different labels.
circles.label.value_counts()

# Visualise the circles.
import matplotlib.pyplot as plt
plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c = y, 
            cmap = plt.cm.RdYlBu)

# Check the shapes of our features and labels
X.shape, y.shape

# View the first example of features and labels.
X_sample = X[0]
y_sample = y[0]
print(f"Values for one sample of x: {X_sample} and the same for y: {y_sample}")
print(f"Shapes for one sample of X: {X_sample.shape} and the same for y: {y_sample.shape}")

## Turn data to tensores and create train and test split.

import torch
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# View the first five samples;
X[:5], y[:5]

# Splot the data into train and test.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state=42)            
len(X_train), len(X_test), len(y_train), len(y_test)

## Bulding the model.
from torch import nn

# Makde device agnositic
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else "cpu"

device

# 1. Construct a model class that subclasses nn.Module
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Create t linear layers capable of handing X and y
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)
        
    # Define a forward pass method containing the FP comuptation.
    def forward(self, x):
        # Return the output of layer 2, a single feature, the same shape as y
        return self.layer_2(self.layer_1(x))
    
# Create and instance of the model and send it to the target device.
model_0 = CircleModelV0().to(device)
model_0

# Make some predictions with the model.
untrained_preds = model_0(X_test.to(device))

print(f"Length of predictionsL {len(untrained_preds)}, Shape: {untrained_preds}")
print(f"Length of test samples: {len(y_test)}, Shape: {y_test.shape}")
print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
print(f"\nFirst 10 test labels:\n{y_test[:10]}")

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()

# Create an optimizer 
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.1)

# Function to calculate the accuracy.
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Lets see what comes out of model after 5 forward passees
y_logits = model_0(X_test.to(device))[:5]
y_logits # Basically random since no training.

# Use sigmoid actication functuion to get probability
y_pred_prob = torch.sigmoid(y_logits)
y_pred_prob

# Find the predicted labels (round the prediction probabilities)
y_preds = torch.round(y_pred_prob)

# In full
y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))

# Check for equality
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))

# Get rid of extra dimension
y_preds.squeeze()

# Train the model.

epochs = 100

# Put data to target device.
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# Build training and evaluation loop
for epoch in range(epochs):
    ### Training
    model_0.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = model_0(X_train).squeeze() # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device 
    y_pred = torch.round(torch.sigmoid(y_logits)) # turn logits -> pred probs -> pred labls
  
    # 2. Calculate loss/accuracy
    loss = loss_fn(y_logits,
                   y_train)
    
    acc = accuracy_fn(y_true=y_train, 
                      y_pred=y_pred) 

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    model_0.eval()
    
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = model_0(X_test).squeeze() 
        test_pred = torch.round(torch.sigmoid(test_logits))
        
        # 2. Caculate loss/accuracy
        test_loss = loss_fn(test_logits,
                            y_test)
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    
import requests
from pathlib import Path 

## Visualise Predictions ###

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

