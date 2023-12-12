import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from functools import partial
import sympy as sp

from generate_bifurcs import *
from scipy.integrate import odeint
from scipy.optimize import linear_sum_assignment
from numpy.polynomial import Polynomial
import pytorch_lightning as pl
from bifurc_transformer import *


def simulate(equation, n=200, n_simul=256):
    args = sp.symbols('x t r')
    f = sp.lambdify(args, equation, 'numpy')
    x0 = np.random.rand(n) * 10 - 5
    t = np.geomspace(0.1, 10.1, n_simul) - 0.1
    r = np.random.rand(n) * 2 - 1
    x = np.zeros((n, n_simul))
    for i in range(n):
        x[i] = odeint(f, x0[i], t, args=(r[i],))[:, 0]
    return np.float32(x), np.float32(r)


class BifurcationPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=5, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # self.transformer = nn.Transformer(d_model=100, nhead=5, num_encoder_layers=3, num_decoder_layers=3,
        #                                   dim_feedforward=100, dropout=0.1, batch_first=True)
        # self.emplacer = nn.Linear(1, 100)

        self.n_features = 256
        self.n_queries = 50

        pos_encoding = PositionalEncoding(self.n_features)
        encoder_layer = PosEncoderLayer(pos_encoding, d_model=self.n_features, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = PosEncoder(encoder_layer, num_layers=4)

        self.queries = nn.Embedding(self.n_queries, self.n_features)
        decoder_layer = PosDecoderLayer(pos_encoding, d_model=self.n_features, nhead=8, dim_feedforward=1024, batch_first=True)
        self.decoder = PosDecoder(decoder_layer, num_layers=4)

        self.final_decoder = nn.Linear(self.n_features, 16)

    def forward(self, x, r):
        # x = self.pos_encoding(x, r)
        # x = self.emplacer(r.unsqueeze(-1)) + x
        # x[..., -1] = r
        mem = self.encoder(x, r)

        queries = self.queries(torch.arange(self.n_queries, device=x.device)).expand(x.shape[0], -1, -1)
        x = self.decoder(queries, mem, r)

        return self.final_decoder(x)

    def loss(self, x, y):
        # x : (batch, 100, 100)
        # y : (batch, (<positions>, segments, 100))
        # loss for the shape

        # determine assignments
        assignments = []
        # cost_matrices = []
        shape_loss, pred_loss, pos_loss, stab_loss = 0, 0, 0, 0
        for i in range(len(x)):
            if len(y[i]) > 0:
                # params = x[i, :, 3:]
                # taylor = create_taylor_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
                # vector = create_bezier_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
                # pred = (params * vector.unsqueeze(-2)).sum(-1).T  # 20 x 100

                pred = x[i, :, 1:].unsqueeze(-2).expand(-1, y[i].shape[0], -1)
                targ = y[i].unsqueeze(0).expand(x.shape[1], -1, -1)
                cost_matrix = F.mse_loss(pred, targ, reduction='none').mean(dim=-1).T
                # xp = x[i, :, 1:5].unsqueeze(0).expand(y[i].shape[0], -1, -1)
                # yp = y[i][:, 1:5].unsqueeze(1).expand(-1, x.shape[1], -1)
                # cost_matrix += F.l1_loss(xp, yp, reduction='none').mean(dim=-1)
                # xs = torch.sigmoid(x[i, :, 5:6]).unsqueeze(0).expand(y[i].shape[0], -1, -1)
                # ys = (y[i][:, 0:1] > 0).float().unsqueeze(1).expand(-1, x.shape[1], -1)
                # cost_matrix += nn.L1Loss(reduction='none')(xs, ys).mean(dim=-1) * 0.01

                assigment = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
                assignments.append(assigment)

                pred = x[i, assigment[1], 1:]
                targ = y[i][assigment[0]]
                stab_loss = F.binary_cross_entropy_with_logits(pred[:, 0], targ[:, 0])
                pos_loss = F.l1_loss(pred[:, 1:5], targ[:, 1:5])
                shape_loss = F.l1_loss(pred[:, 5:], targ[:, 5:])

                # curve_loss += F.mse_loss(x[i, assigment[1], 1:], y[i][assigment[0]])
            else:
                assignments.append(([], []))
        shape_loss /= len(x)
        pos_loss /= len(x)
        stab_loss /= len(x)

        # loss for the shape
        # shape_loss = []
        # for i, a in enumerate(assignments):
        #     # params = x[i, a[1], 3:]
        #     if y[i].numel() == 0 or len(a[0]) == 0:
        #         shape_loss.append(torch.tensor(0, device=x.device))
        #         continue
        #     # taylor = create_taylor_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
        #     # vector = create_bezier_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
        #     shape_loss_i = F.l1_loss(x[i, a[1], 6:], y[i][:, 7:-1])
        #     if shape_loss_i < 2:
        #         shape_loss.append(shape_loss_i)
        #     else:
        #         assignments[i] = ([], [])
        # shape_loss = torch.stack(shape_loss).mean()
        # shape_loss = 0

        # loss for the segment prediction
        pred = x[..., 0]
        true = torch.zeros_like(pred).bool()
        for i, a in enumerate(assignments):
            true[i, a[1]] = True
        true = torch.zeros_like(x[..., 0])
        for i, a in enumerate(assignments):
            true[i, a[1]] = 1
        # pred_loss = sigmoid_focal_loss(x[..., 0], true, alpha=0.25, gamma=2, reduction='mean')
        pred_loss = 0.5 * (nn.BCEWithLogitsLoss()(pred[true], torch.ones_like(pred[true])) +
                     nn.BCEWithLogitsLoss()(pred[~true], torch.zeros_like(pred[~true])))

        # # loss for the position prediction
        # pos_loss = []
        # for i, a in enumerate(assignments):
        #     if y[i].numel() == 0 or len(a[0]) == 0:
        #         pos_loss.append(torch.tensor(0, device=x.device))
        #         continue
        #     pos_loss.append(nn.L1Loss()(x[i, a[1], 1:5], y[i][:, 1:5]))
        # pos_loss = torch.stack(pos_loss).mean()

        # # loss for the stability prediction
        # stab_loss = []
        # for i, a in enumerate(assignments):
        #     if y[i].numel() == 0 or len(a[0]) == 0:
        #         stab_loss.append(torch.tensor(0, device=x.device))
        #         continue
        #     stab_loss.append(nn.BCEWithLogitsLoss()(x[i, a[1], 5], y[i][:, 0]))
        # stab_loss = torch.stack(stab_loss).mean()

        return shape_loss, pred_loss, pos_loss, stab_loss
        # return curve_loss, pred_loss#, pos_loss, stab_loss

    def training_step(self, batch, batch_idx):
        simul, param, bifurcation = batch

        # random trajectory dropout
        if self.current_epoch > 0:
            match np.random.rand():
                case x if x > 0.8:
                    mask = np.random.rand(simul.shape[1]) > np.random.rand()
                    simul, param = simul[:, mask], param[:, mask]
                case x if 0.6 < x < 0.8:
                    bounds = sorted(np.random.rand(2))
                    mask = torch.logical_and(param[0, :] > bounds[0], param[0, :] < bounds[1])
                    simul, param = simul[:, mask], param[:, mask]
                case x if 0.2 < x < 0.6:
                    simul += torch.randn_like(simul) * torch.rand() * 0.5
                    param += torch.randn_like(param) * torch.rand() * 0.5
            # case x if 0.5 > x > 0.25:
            #     bounds = sorted(np.random.rand(2))
            #     mask = torch.logical_and(simul[-1, :] > bounds[0], simul[-1, :] < bounds[1])
            #     simul, param = simul[:, mask], param[:, mask]

        x = self(simul, param)
        # curve_loss, pred_loss = self.loss(x, bifurcation)
        shape_loss, pred_loss, pos_loss, stab_loss = self.loss(x, bifurcation)
        loss = shape_loss + pred_loss + pos_loss + stab_loss
        # loss = curve_loss + pred_loss
        # self.log("curve_loss", curve_loss, prog_bar=True)
        self.log("shape_loss", shape_loss, prog_bar=True)
        self.log("pred_loss", pred_loss, prog_bar=True)
        self.log("pos_loss", pos_loss, prog_bar=True)
        self.log("stab_loss", stab_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)


class BifurcationData(Dataset):
    def __init__(self, simul_data, par_data, curve_index_data, curve_data):
        self.curve_index_data = torch.asarray(curve_index_data, dtype=torch.int)
        self.curve_data = torch.asarray(curve_data, dtype=torch.float32)
        self.simul_data = torch.asarray(simul_data, dtype=torch.float32)
        self.par_data = torch.asarray(par_data, dtype=torch.float32)

    def __len__(self):
        return len(self.curve_index_data)

    def __getitem__(self, idx):
        bifurc_curves = self.curve_data[self.curve_index_data[idx][0]:self.curve_index_data[idx][1]]
        par_datum = self.par_data[idx]
        sort_mask = np.argsort(par_datum)
        return self.simul_data[idx][sort_mask], self.par_data[idx][sort_mask], bifurc_curves


def bifurcation_collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]


def create_bezier_vector(x0, x1, n=100):
    x = torch.asarray(np.linspace(x0.cpu() * 0, x1.cpu() * 0 + 1, n)).float().to(x0.device)
    return torch.stack([(1 - x) ** 5,
                        5 * x * (1 - x) ** 4,
                        10 * x ** 2 * (1 - x) ** 3,
                        10 * x ** 3 * (1 - x) ** 2,
                        5 * x ** 4 * (1 - x),
                        x ** 5], dim=-1)


if __name__ == '__main__':
    try:
        saved_data = np.load('dataset_500.npz')
        data = [saved_data[key] for key in saved_data]
    # dataset = pickle.load(open('dataset.pkl', 'rb'))
    # dataset = pickle.load(open(r"C:\Users\Lukas\dataset.pkl", 'rb'))
    except FileNotFoundError:
        # bases = ['a01', 'a02*r',
        #          'a03*(x+b/4)', 'a04*r*(x+b/4)', 'a05*(x+b/4)**2', 'a06*r*(x+b/4)**2', 'a06*(x+b/4)**3', 'a07*r*(x+b/4)**3',
        #          'a08*sin(2*c01*(x+b/4))', 'a09*r*sin(2*c02*(x+b/4))',
        #          'a10*sin(2*c03*r*(x+b/4))', 'a11*r*sin(2*c04*r*(x+b/4))']
        bases = ['a01', 'a02*r',
                 'a03*(x+b/2)', 'a04*r*(x+b/2)', 'a05*(x+b/2)**2', 'a06*r*(x+b/2)**2', 'a06*(x+b/2)**3', 'a07*r*(x+b/2)**3']
        # params = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10',  a11',
        #           'b', 'c01', 'c02', 'c03', 'c04']
        params = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'b']
        equations = generate_equations(bases, params, append='-0.01*x**5') * 100
        print('Starting data generation')
        equations, curve_index_data, curve_data = generate_data(equations, mesh_size=1000, use_mp=True)
        # curve_index_data, curve_data = pickle.load(open('curve_data.pkl', 'rb'))

        map_func = partial(simulate, n=200, n_simul=256)
        # data = map(map_func, tqdm(equations, desc='Simulating systems'))
        with WorkerPool(n_jobs=12) as pool:
            data = pool.map(map_func, equations,
                            progress_bar=True, progress_bar_options={'desc': 'Simulating systems'})
        # for i, eq in enumerate(tqdm(equations, desc='Simulating systems')):
        #     data.append(simulate(eq, n=200, n_simul=400))
        #     # if (i + 1) % 10 == 0: print('s', i + 1)
        simul_data, par_data = list(zip(*data))
        simul_data, par_data = np.stack(simul_data), np.stack(par_data)
        data = (simul_data, par_data, curve_index_data, curve_data)
        np.savez('dataset.npz', *data)
        equations = [sp.pretty(eq, use_unicode=False) for eq in equations]
        pickle.dump(equations, open('equations.pkl', 'wb'))


    dataset = BifurcationData(*data)
    dataloader = DataLoader(dataset, batch_size=20, collate_fn=bifurcation_collate, num_workers=6, shuffle=True)

    # model = BifurcationPredictor()
    model = BifurcationPredictor().load_from_checkpoint('lightning_logs/version_11/epoch=295-step=377400.ckpt')
    trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', gradient_clip_val=0.1)
    trainer.fit(model, dataloader)
