import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.autograd import Variable

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns


from scipy.stats import ortho_group

import psgd_torch as psgd



###### Problem dimensions and data generation ######

d = 1000
n = 100

# mixture of two weibull distributions
mixture = d//2

u = ortho_group.rvs(d)
diag = np.diag(np.random.weibull(43, d))
diag[mixture:, mixture:] = np.random.weibull(37, mixture)

D = np.sqrt(diag)
sqrtSigma1 = u.dot(D).dot(u.T)

shape, scale = 2., 1. # mean and width
u = ortho_group.rvs(d)
diag = np.diag(np.random.weibull(12, d))
diag[mixture:, mixture:] = np.random.weibull(92, mixture)

D = np.sqrt(diag)
sqrtSigma2 = u.dot(D).dot(u.T)


X_tilde = np.random.normal(0, 1 / math.sqrt(d) , (d,n))
mu_1 = np.random.normal(0, 1 / math.sqrt(d), d)
eps = 0.1
v = np.random.normal(0, 1 / math.sqrt(d), d)
mu_2 = math.sqrt(1 - eps ** 2) * mu_1 + eps * v
X = np.zeros((d,n))
y = np.ones(n)
for i in range(n // 2 + 1):
    X[:,i] = sqrtSigma1 @ X_tilde[:,i] + mu_1
for i in range(n // 2, n):
    X[:,i] = sqrtSigma2 @ X_tilde[:,i] + mu_2
    y[i] = -1
X_t = np.transpose(X)


class linearRegression(torch.nn.Module):
    def __init__(self, d):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(d, 1, bias = False)

    def forward(self, x):
        out = self.linear(x).squeeze()
        return out


# Batching the data

class data_set(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data_point = self.data[idx]
        sample = {"data": data_point, "label": label}
        return sample

dataset = data_set(X_t, y)
batch_size = 100
# implementing dataloader on the dataset and printing per batch
trainloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
for i, batch in enumerate(trainloader):
    print(i, batch)



def empirical_generalization_error(d, mu_1, mu_2, num, w, sqrtSigma1,sqrtSigma2):
    error = 0
    for i in range(num // 2):
        v = sqrtSigma1 @ np.random.normal(0, 1 / math.sqrt(d), d)
        v += mu_1
        if np.dot(v, w) < 0:
            error += 1 / num
    for i in range(num // 2, num):
        v = sqrtSigma2 @ np.random.normal(0, 1 / math.sqrt(d), d)
        v += mu_2
        if np.dot(v, w) > 0:
            error += 1 / num
    return error


###### SGD ######
print("Starting SGD")
print("-"*10)

num_epochs = 2000
num_classes = 2
learning_rate = 1

criterion = nn.MSELoss()
model = linearRegression(d)
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Training
sgd_mse = []
sgd_generalization_error = []
for epoch in range(num_epochs):
    for i, batch in enumerate(trainloader):

    
        images = batch["data"].cuda()
        labels = batch["label"].cuda()
        images = Variable(images).float()
        labels = Variable(labels).float()


        optimizer.zero_grad()

        # Run the forward pass
        outputs = model(images).float()
        loss = criterion(outputs, labels)


        # clip grads for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss.backward()
        optimizer.step()
        sgd_mse.append(loss.detach().cpu().item())

    # every 100 epochs
    if epoch % 100 == 0:
      w = torch.nn.utils.parameters_to_vector(model.parameters())
      w_np = w.detach().cpu().numpy()

      # error if nan weights
      if np.isnan(w_np).any():
        print('NaN weights')
        break      

      num = 10 * d
      ge = empirical_generalization_error(d, mu_1, mu_2, num, w_np,sqrtSigma1,sqrtSigma2)
      sgd_generalization_error.append(ge)
      print('Generalization Error SGD: ', ge)
print('Finished Training SGD on Isotropic Eigenvalues')




###### PSGD ######
print("Starting PSGD")
print("-"*10)

criterion = nn.MSELoss()
model = linearRegression(d)
model = model.cuda()
optimizer = psgd.Newton(model.parameters(),
                         lr_params=1.5,
                         lr_preconditioner=.5,
                         preconditioner_update_probability=1,
                         grad_clip_max_norm=1)


# Training
psgd_mse = []
psgd_generalization_error = []
for epoch in range(num_epochs):
    for i, batch in enumerate(trainloader):

        images = batch["data"].cuda()
        labels = batch["label"].cuda()
        images = Variable(images).float()
        labels = Variable(labels).float()
        # Run the forward pass
        outputs = model(images).float()

        loss = criterion(outputs, labels)
        def closure():
          return loss

        # Backprop and perform
        optimizer.step(closure)
        psgd_mse.append(loss.detach().cpu().item())

    # every 100 epochs
    if epoch % 100 == 0:
      w = torch.nn.utils.parameters_to_vector(model.parameters())
      w_np = w.detach().cpu().numpy()
      
      # error if nan weights
      if np.isnan(w_np).any():
        print('NaN weights')
        break


      num = 10 * d
      ge = empirical_generalization_error(d, mu_1, mu_2, num, w_np,sqrtSigma1,sqrtSigma2)
      psgd_generalization_error.append(ge)
      print('Generalization Error PSGD: ', ge)
    optimizer.lr_params *= (0.1) ** (1 / (num_epochs-1))
    # optimizer.lr_preconditioner *= (0.1) ** (1 / (num_epochs-1))

print('Finished Training PSGD on Isotropic Eigenvalues')




sns.set_theme(style="darkgrid", palette="dark")
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot generalization error
sns.lineplot(data=psgd_generalization_error, ax=axes[0], label="PSGD")
sns.lineplot(data=sgd_generalization_error, ax=axes[0], label="SGD")
axes[0].set_title("Generalization Error")
axes[0].set_xlabel("Epochs (x100)")
axes[0].set_ylabel("Error")
axes[0].legend()

# Plot MSE
sns.lineplot(data=psgd_mse, ax=axes[1], label=f"PSGD MSE: {psgd_mse[-1]}")
sns.lineplot(data=sgd_mse, ax=axes[1], label=f"SGD MSE: {sgd_mse[-1]:.0e}")

# ylim
axes[1].set_ylim(-1e-2, 0.01)
axes[1].set_title("MSE")
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("MSE")
axes[1].legend()

plt.tight_layout()
plt.savefig("psgd_sgd_weibull_comparison.png")
plt.show()
