import os

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
from scipy import io as sio

from generate_bifurcs import *
from scipy.integrate import odeint
from scipy.optimize import linear_sum_assignment
from numpy.polynomial import Polynomial
import pytorch_lightning as pl
import bifurc_transformer as bt


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


def decode_trajectories(bboxes, branches):
    branches = branches * (bboxes[:, 3:4] - bboxes[:, 2:3]) + bboxes[:, 2:3]
    branches = F.pad(branches.unsqueeze(-1), (1, 0))
    for i in range(len(branches)):
        branches[i, :, 0] = torch.linspace(bboxes[i, 0], bboxes[i, 1], len(branches[i]))
    return branches


class BifurcationPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=5, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # self.transformer = nn.Transformer(d_model=100, nhead=5, num_encoder_layers=3, num_decoder_layers=3,
        #                                   dim_feedforward=100, dropout=0.1, batch_first=True)
        # self.emplacer = nn.Linear(1, 100)

        self.n_features = 128
        self.n_queries = 50

        pos_encoding = bt.PositionalEncoding(self.n_features)
        encoder_layer = bt.PosEncoderLayer(pos_encoding, d_model=self.n_features, nhead=8, dim_feedforward=1024,
                                        batch_first=True)
        self.encoder = bt.PosEncoder(encoder_layer, num_layers=4)

        self.queries = nn.Embedding(self.n_queries, self.n_features)
        decoder_layer = bt.PosDecoderLayer(pos_encoding, d_model=self.n_features, nhead=8, dim_feedforward=1024,
                                        batch_first=True)
        self.decoder = bt.PosDecoder(decoder_layer, num_layers=4)

        self.conf_decoder = nn.Linear(self.n_features, 1)
        self.bbox_decoder = nn.Linear(self.n_features, 4)
        self.stab_decoder = nn.Linear(self.n_features, 1)
        self.branch_decoder = nn.Linear(self.n_features, 101)

    def forward(self, r, x):
        # x = self.pos_encoding(x, r)
        # x = self.emplacer(r.unsqueeze(-1)) + x
        # x[..., -1] = r
        mem = self.encoder(x, r)

        queries = self.queries(torch.arange(self.n_queries, device=x.device)).expand(x.shape[0], -1, -1)
        x = self.decoder(queries, mem, r)

        return self.conf_decoder(x), self.bbox_decoder(x), self.stab_decoder(x), self.branch_decoder(x)

    def loss(self, x, y):
        # x : (batch, 100, 100)
        # y : (batch, (<positions>, segments, 100))
        # loss for the shape

        conf_b, bbox_b, stab_b, branch_b = x
        # cost_matrices = []
        conf_loss, bbox_loss, stab_loss, branch_loss = 0., 0., 0., 0.
        for i in range(len(x)):
            if len(y[i]) == 0:
                conf_loss += nn.BCEWithLogitsLoss()(conf_b[i], torch.zeros_like(conf_b[i]))
            else:
                bbox, stab, branch = y[i]

                # # decode trajectories into bifurcation diagram
                # pred_traj = decode_trajectories(bbox_b[i].detach(), branch_b[i].detach())
                # true_traj = decode_trajectories(bbox, branch)
                # n_pred, n_true = len(pred_traj), len(true_traj)
                # pred_traj = pred_traj.unsqueeze(0).expand(n_true, n_pred, -1, -1)
                # true_traj = true_traj.unsqueeze(1).expand(n_true, n_pred, -1, -1)

                # calculate component losses
                n_pred, n_true = len(bbox_b[i]), len(bbox)
                pred_bbox = bbox_b[i].detach().unsqueeze(0).expand(n_true, n_pred, -1)
                true_bbox = bbox.unsqueeze(1).expand(n_true, n_pred, -1)
                pred_stab = stab_b[i].detach().unsqueeze(0).expand(n_true, n_pred, -1)
                true_stab = stab.unsqueeze(1).unsqueeze(2).expand(n_true, n_pred, -1)
                pred_branch = branch_b[i].detach().unsqueeze(0).expand(n_true, n_pred, -1)
                true_branch = branch.unsqueeze(1).expand(n_true, n_pred, -1)

                # calculate cost matrix as pairwise mse between trajectories
                # traj_dist = F.mse_loss(pred_traj, true_traj, reduction='none').sum(dim=-1).mean(dim=-1)
                bbox_dist = F.mse_loss(pred_bbox, true_bbox, reduction='none').mean(dim=-1)
                stab_dist = F.mse_loss(pred_stab, true_stab, reduction='none').mean(dim=-1)
                branch_dist = F.mse_loss(pred_branch, true_branch, reduction='none').mean(dim=-1)
                dist = stab_dist + bbox_dist + 0.1 * branch_dist
                # dist = 0.5 * stab_dist + bbox_dist
                assign_true, assign_pred = linear_sum_assignment(dist.detach().cpu().numpy())

                # use assignment to calculate bifurcation losses
                bbox_loss += F.mse_loss(bbox_b[i, assign_pred], bbox[assign_true])
                stab_loss += F.mse_loss(stab_b[i, assign_pred].view(-1), stab[assign_true])
                branch_loss += F.mse_loss(branch_b[i, assign_pred], branch[assign_true])

                # use assignment to calculate balanced confidence loss
                true_preds = torch.zeros_like(conf_b[i]).bool()
                true_preds[assign_pred] = True
                pos_conf_loss = F.binary_cross_entropy_with_logits(conf_b[i][true_preds], torch.ones_like(conf_b[i][true_preds]))
                neg_conf_loss = F.binary_cross_entropy_with_logits(conf_b[i][~true_preds], torch.zeros_like(conf_b[i][~true_preds]))
                conf_loss += 0.5 * (pos_conf_loss + neg_conf_loss)

        return conf_loss, bbox_loss, stab_loss, branch_loss

    def training_step(self, batch, batch_idx):
        param, simul, bifurcation = batch

        # random trajectory dropout
        # if self.current_epoch > 0:
        #     match np.random.rand():
        #         case x if x > 0.7:
        #             mask = np.random.rand(simul.shape[1]) > np.random.rand()
        #             simul, param = simul[:, mask], param[:, mask]
        #         # case x if 0.6 < x < 0.8:
        #         #     bounds = sorted(np.random.rand(2))
        #         #     mask = torch.logical_and(param[0, :] > bounds[0], param[0, :] < bounds[1])
        #         #     simul, param = simul[:, mask], param[:, mask]
        #         case x if 0.4 < x < 0.7:
        #             simul += torch.randn_like(simul) * torch.rand(1).to(self.device) * 0.5
        #             param += torch.randn_like(param) * torch.rand(1).to(self.device) * 0.5
            # case x if 0.5 > x > 0.25:
            #     bounds = sorted(np.random.rand(2))
            #     mask = torch.logical_and(simul[-1, :] > bounds[0], simul[-1, :] < bounds[1])
            #     simul, param = simul[:, mask], param[:, mask]

        x = self(param, simul)
        # curve_loss, pred_loss = self.loss(x, bifurcation)
        conf_loss, bbox_loss, stab_loss, branch_loss = self.loss(x, bifurcation)

        # regularize losses
        reg_conf_loss = conf_loss * torch.minimum(((bbox_loss + stab_loss + branch_loss) / 2).detach(),
                                                  torch.ones(1).to(self.device))
        loss = reg_conf_loss + bbox_loss + stab_loss + branch_loss

        # loss = curve_loss + pred_loss
        # self.log("curve_loss", curve_loss, prog_bar=True)
        self.log("conf_loss", conf_loss, prog_bar=True)
        self.log("bbox_loss", bbox_loss, prog_bar=True)
        self.log("stab_loss", stab_loss, prog_bar=True)
        self.log("branch_loss", branch_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)


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


class BifurcationDataMathematica(Dataset):
    def __init__(self, file_name=None, n=300):
        self.params, self.simuls, self.bifurcs = [], [], []
        if file_name is None:
            for file_idx in tqdm(range(1, n + 1), desc='Loading data'):
                params, simuls, bifurcs = self.load_data(f'data/mat_files/data_{file_idx}.mat')
                self.params.append(params)
                self.simuls.append(simuls)
                self.bifurcs.extend(bifurcs)
            self.simuls = torch.from_numpy(np.concatenate(self.simuls)).float()
            self.params = torch.from_numpy(np.concatenate(self.params)).float()
        else:
            self.params, self.simuls, self.bifurcs = self.load_data(file_name, safe=True)
            self.simuls = torch.from_numpy(self.simuls).float()
            self.params = torch.from_numpy(self.params).float()

    def load_data(self, file_name, safe=False):
        bifurcs, simuls = sio.loadmat(file_name)['Expression1']
        params = np.stack([np.stack(s[0]) for s in simuls]).squeeze()
        simuls = np.stack([np.stack(s[1]) for s in simuls]).squeeze()
        if safe:
            bifurcs = [np.stack(b) for b in bifurcs]
        for i in range(len(bifurcs)):
            bifurcs[i] = [torch.from_numpy(np.stack(b)).float() for b in bifurcs[i].T]
            if len(bifurcs[i]) > 0:
                bifurcs[i][1] = bifurcs[i][1].ravel()
        return params, simuls, bifurcs

    def __len__(self):
        return len(self.simuls)

    def __getitem__(self, idx):
        return self.params[idx], self.simuls[idx], self.bifurcs[idx]


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
    dataset = BifurcationDataMathematica()
    dataloader = DataLoader(dataset, batch_size=20, collate_fn=bifurcation_collate, num_workers=0, shuffle=True)

    # model = BifurcationPredictor()
    model = BifurcationPredictor().load_from_checkpoint('lightning_logs/version_18/checkpoints/epoch=31-step=64960.ckpt')
    trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', gradient_clip_val=0.1)
    trainer.fit(model, dataloader)
