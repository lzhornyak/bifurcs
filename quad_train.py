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


def subcritical(x, t, r):
    return r * x - x ** 3


def solve(x0, r, t=10):
    t = np.geomspace(0.01, t, 100)
    sol = integrate.odeint(subcritical, x0, t, args=(r,))
    return sol


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hsize = 16
        self.net = nn.ModuleList([
            QuadLayer(2, self.hsize),
            QuadLayer(self.hsize, 2)
        ])
        # self.mask = torch.ones(self.hsize, dtype=torch.bool)

    # def set_mask(self):
    #     self.mask = torch.rand(self.hsize) > 0.2

    def forward(self, x):
        x = self.net[0](x)
        # if self.training:
        #     x[..., ~self.mask] = 0.
        x = self.net[1](torch.tanh(x))
        return x

class Learner(pl.LightningModule):
    def __init__(self, t_span, model, sub_model):
        super().__init__()
        self.sub_model = sub_model
        self.model, self.t_span = model, t_span

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        initial, sol = batch
        # self.sub_model.set_mask()
        t_eval, y_hat = self.model(initial, self.t_span)
        loss = nn.MSELoss()(y_hat[..., 0], sol.T)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "loss"}]


if __name__ == '__main__':
    n = 1000
    t = 5
    x0 = np.stack([random.normal(0, 2, n), random.normal(0, 1, n)])  # x0, r
    sol = []
    for i in range(n):
        sol.append(solve(x0[0, i], x0[1, i], t=t))

    mask = [s is not None for s in sol]
    sol = np.asarray([s for s in sol if s is not None])
    x0 = x0[:, mask]
    print(len(sol))

    dataset = TensorDataset(torch.from_numpy(x0.T).float(), torch.from_numpy(sol).float().squeeze())
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    f = Network().to(device)
    model = NeuralODE(f, sensitivity='adjoint', solver='tsit5', interpolator=None, atol=1e-3, rtol=1e-3).to(device)

    t_span = torch.from_numpy(np.geomspace(0.01, t, 100)).float()
    learn = Learner(t_span, model, f)
    trainer = pl.Trainer(max_epochs=250, accelerator="cpu", devices=1)
    trainer.fit(learn, dataloader)

    coords = torch.stack(torch.meshgrid([torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100)])).permute(1, 2, 0)
    ts = torch.jit.trace_module(f.eval().to(torch.double).cpu(), {'forward': coords[0, 0].to(torch.double)})
    ys = torch.jit.freeze(ts)
    ts.save('model.pt')
