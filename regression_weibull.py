#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import math
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ortho_group


import psgd_torch as psgd


# Problem dimensions and data generation
d = 1000
n = 100
sig = 0.1  # Standard deviation of noise

# mixture of two weibull distributions
mixture = d//2

diag = np.diag(np.random.weibull(2, d))
diag[mixture:, mixture:] = np.random.weibull(6, mixture)

D = np.sqrt(diag)
u = ortho_group.rvs(d)
sqrtSigma = u.dot(D).dot(u.T)

wst = np.random.normal(0, 1, d)
X = np.random.normal(0, 1, (n, d)) @ sqrtSigma
z = sig * np.random.normal(0, 1, n)
y = X @ wst + z

class linearRegression(nn.Module):
    def __init__(self, d):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        return self.linear(x).squeeze()

    def to_device(self, device):
        self.to(device)

# Dataset and DataLoader setup
class data_set(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"data": self.data[idx], "label": self.labels[idx]}

dataset = data_set(X, y)  # Updated to use new X and y directly
trainloader = DataLoader(dataset, batch_size=1024, shuffle=True)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# set initilization of linearRegression to orthogonal initalization

smallLinear = linearRegression(d)
smallLinear.to_device(device)  # Move model to the appropriate device
# torch.nn.init.normal_(smallLinear.linear.weight, mean=0, std=0.01)
torch.nn.init.orthogonal_(smallLinear.linear.weight, gain=1)
num_epochs = 2_000_000
###### SGD ######
print("Starting SGD")
print("-"*10)

# Model, Loss, and Optimizer
model = copy.deepcopy(smallLinear)
criterion = nn.MSELoss()

optimizer = optim.SGD(model.parameters(), lr=1e-6)

sgd_mse = []
# Training Loop
for epoch in range(num_epochs):
    for batch in trainloader:
        images = Variable(batch["data"]).float().to(device)  # Move images to device
        labels = Variable(batch["label"]).float().to(device)  # Move labels to device

        optimizer.zero_grad()
        outputs = model(images) 
        loss = criterion(outputs, labels)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        loss.backward()
        optimizer.step()


    if epoch % 10_000 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')
        sgd_mse.append(loss.detach().cpu().item())

        

print('Training complete')

# %%
###### PSGD ######
print("Starting PSGD")
print("-"*10)


# Model, Loss, and Optimizer
model = copy.deepcopy(smallLinear)
criterion = nn.MSELoss()
optimizer = psgd.Newton(model.parameters(),
                         lr_params=1,
                         lr_preconditioner=1,
                         preconditioner_update_probability=1,
                         grad_clip_max_norm=1)

psgd_mse = []
# Training Loop
for epoch in range(num_epochs):
    for batch in trainloader:
        images = Variable(batch["data"]).float().to(device)  # Move images to device
        labels = Variable(batch["label"]).float().to(device)  # Move labels to device

        outputs = model(images)
        loss = criterion(outputs, labels)
        def closure():
            return loss
        optimizer.step(closure)

    if epoch % 10_000 == 0:
        print(f'Epoch {epoch}: Loss = {loss.item()}')
        psgd_mse.append(loss.detach().cpu().item())
    optimizer.lr_params *= (0.1) ** (1 / (num_epochs//100-1))
    optimizer.lr_preconditioner *= (0.1) ** (1 / (num_epochs//100-1))

print('Training complete')

#%%
sns.set_theme(style="darkgrid", palette="muted")
# Create subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot MSE using semilogarithmic scale on the y-axis
sns.lineplot(x=np.arange(len(psgd_mse)), y=psgd_mse, ax=ax, label="PSGD MSE")
sns.lineplot(x=np.arange(len(sgd_mse)), y=sgd_mse, ax=ax, label="SGD MSE")
ax.set_yscale('log')  # Set the y-axis to a logarithmic scale

ax.set_title("Mean Squared Error Comparison")
ax.set_xlabel("Iterations (x 10,000)")
ax.set_ylabel("Log MSE")
ax.legend()

plt.tight_layout()
plt.savefig("mse_comparison_log_weibull.png")
plt.show()