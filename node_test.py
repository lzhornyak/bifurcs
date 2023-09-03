from torchdyn.core import NeuralODE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from scipy import ndimage

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# system to solve
def pitchfork(x, t, r):
    return r * x - x ** 3

# generate data
def solve(x0, r):
    t = np.linspace(0, 10, 100)
    sol = integrate.odeint(subcritical, x0, t, args=(r,))
    return sol

x0 = np.stack([random.uniform(-1, 1, 100), random.uniform(-1, 1, 100)]) #x0, r
sol = []
for i in range(100):
    sol.append(solve(x0[0, i], x0[1, i]))

mask = [s is not None for s in sol]
sol = np.asarray([s for s in sol if s is not None])
x0 = x0[:, mask]

# train model
dataset = TensorDataset(torch.from_numpy(x0.T).float(), torch.from_numpy(sol).float().squeeze())
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

f = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 2))
model = NeuralODE(f, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(device)

class Learner(pl.LightningModule):
    def __init__(self, t_span, model):
        super().__init__()
        self.model, self.t_span = model, t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x0, sol = batch
        t_eval, y_hat = self.model(x0, self.t_span)
        loss = nn.MSELoss()(y_hat[..., 0], sol.T)
        loss += nn.MSELoss()(y_hat[..., 1], x0[:, 1].expand_as(y_hat[..., 1]))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

t_span = torch.linspace(0, 10, 100)
learn = Learner(t_span, model)
trainer = pl.Trainer(min_epochs=200, max_epochs=250)
trainer.fit(learn, dataloader)

