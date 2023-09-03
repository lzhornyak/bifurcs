from torchdyn.core import NeuralODE
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt
from numpy import random
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class QuadLayer(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.l1 = nn.Linear(in_features + 1, out_features, **kwargs)
        self.l2 = nn.Linear(in_features + 1, out_features, **kwargs)
        self.ls = nn.Linear(in_features + 1, out_features, **kwargs)

    def forward(self, x):
        x = nn.functional.pad(x, (0, 1), "constant", 1.0)
        l1 = self.l1(x)
        l2 = self.l2(x)
        ls = self.ls(x)
        return l1 * l2 + ls

def lorentz(x, t, r):
    return np.array([r[0] * (x[1] - x[0]), x[0] * (r[1] - x[2]) - x[1], x[0] * x[1] - r[2] * x[2]])

def solve(x0, r, n=1000):
    t = np.linspace(0.0, 1, n)
    sol = integrate.odeint(lorentz, x0, t, args=(r,))
    return sol

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsize = 32
        self.net = nn.ModuleList([
            QuadLayer(6, self.hsize),
            QuadLayer(self.hsize, 6)
        ])

    def forward(self, x):
        x = self.net[0](x)
        x = self.net[1](torch.tanh(x))
        return x

class Learner(pl.LightningModule):
    def __init__(self, t_span, model):
        super().__init__()
        self.model, self.t_span = model, t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        initial, sol = batch
        t_eval, y_hat = self.model(initial, self.t_span)
        loss = nn.MSELoss()(y_hat[..., :3], sol.transpose(0, 1))
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss"}]

if __name__ == '__main__':

    n = int(5e4)
    ns = 100
    max_val = 1e2
    x0 = np.concatenate([random.random((n * 5, 3)) * 20 - 10,
                         random.random((n * 5, 3)) * 10], axis=1)

    s, r, b = x0[:, 3], x0[:, 4], x0[:, 5]
    stability = s * (s + b + 3) / (s - b - 1)
    x0 = x0[stability > r][:n]

    sol = []
    for i in range(n):
        sol.append(solve(x0[i, :3], x0[i, 3:], n=ns))
        if (i + 1) % 1000 == 0:
            print('done ', i + 1)

    # for i in range(n):
    #     plt.plot(sol[i][:, 0], sol[i][:, 1])
    #     plt.show()
    #
    # raise TypeError

    sol = np.asarray(sol)
    mask = np.array([np.abs(s).max() < max_val for s in sol])
    sol = sol[mask]
    x0 = x0[mask]

    bs = 10000
    dataset = TensorDataset(torch.from_numpy(x0).float(), torch.from_numpy(sol).float().squeeze())
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    f = Network().to(device)
    model = NeuralODE(f, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(device)

    t_span = torch.from_numpy(np.linspace(0.0, 1.0, ns)).float().to(device)
    learn = Learner(t_span, model)
    trainer = pl.Trainer(max_epochs=1000, accelerator='auto', devices=1)
    trainer.fit(learn, dataloader)

    sample = next(iter(dataloader))[0]
    ts = torch.jit.trace_module(f.eval().to(torch.double).cpu(), {'forward': sample.to(torch.double).cpu()})
    ys = torch.jit.freeze(ts)
    ts.save('model.pt')
