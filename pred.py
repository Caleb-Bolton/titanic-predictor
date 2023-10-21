import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from helper import *
from SimpleNN import SimpleNN
import numpy as np
import matplotlib.pyplot as plt


X, y = preprocess_titanic_dataset('train.csv')
X, y = convert_to_tensors(X, y)

input_dim = X.shape[1]
output_dim = 1

model = SimpleNN([input_dim, 50, 50, 50, 50, 50, output_dim])
model.load_state_dict(torch.load('model.pth'))
model.eval()
with torch.no_grad():
    output_tensor = model(X)
print(np.round(output_tensor.numpy()))
