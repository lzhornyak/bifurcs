import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle

from generate_bifurcs import *
from scipy.integrate import odeint
from scipy.optimize import linear_sum_assignment
from numpy.polynomial import Polynomial
import pytorch_lightning as pl
from bifurc_transformer import *


def simulate(equation, n=100):
    f = eval('lambda x, t, r: ' + equation)
    x0 = np.random.rand(n) * 10 - 5
    t = np.geomspace(0.1, 10.1, 100) - 0.1
    r = np.random.rand(n) * 2 - 1
    x = odeint(f, x0, t, args=(r,))
    return x.T, r


class BifurcationPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=5, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # self.transformer = nn.Transformer(d_model=100, nhead=5, num_encoder_layers=3, num_decoder_layers=3,
        #                                   dim_feedforward=100, dropout=0.1, batch_first=True)
        # self.emplacer = nn.Linear(1, 100)

        pos_encoding = PositionalEncoding(100)
        encoder_layer = PosEncoderLayer(pos_encoding, d_model=100, nhead=5, dim_feedforward=500, batch_first=True)
        self.encoder = PosEncoder(encoder_layer, num_layers=3)

        self.queries = nn.Embedding(20, 100)
        decoder_layer = PosDecoderLayer(pos_encoding, d_model=100, nhead=5, dim_feedforward=500, batch_first=True)
        self.decoder = PosDecoder(decoder_layer, num_layers=3)

        self.final_decoder = nn.Linear(100, 13)

    def forward(self, x, r):
        # x = self.pos_encoding(x, r)
        # x = self.emplacer(r.unsqueeze(-1)) + x
        # x[..., -1] = r
        mem = self.encoder(x, r)

        queries = self.queries(torch.arange(20, device=x.device)).expand(x.shape[0], -1, -1)
        x = self.decoder(queries, mem, r)

        return self.final_decoder(x)

    def loss(self, x, y):
        # x : (batch, 100, 100)
        # y : (batch, (<positions>, segments, 100))
        # loss for the shape

        # determine assignments
        assignments = []
        # cost_matrices = []
        for i in range(len(x)):
            if len(y[i][1]) > 0:
                params = x[i, :, 3:]
                taylor = create_taylor_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
                pred = (params * taylor.unsqueeze(-2)).sum(-1).T  # 20 x 100

                targ = y[i][1].unsqueeze(0).expand(pred.shape[0], -1, -1)
                cost_matrix = nn.MSELoss(reduction='none')(pred, targ).mean(dim=-1).T
                xp = x[i, :, 1:3].unsqueeze(0).expand(y[i][0].shape[0], -1, -1)
                yp = y[i][0].unsqueeze(1).expand(-1, x.shape[1], -1)
                cost_matrix += nn.L1Loss(reduction='none')(xp, yp).mean(dim=-1)

                assignments.append(linear_sum_assignment(cost_matrix.detach().cpu().numpy()))
                # cost_matrices.append(cost_matrix)
            else:
                assignments.append(([], []))

        # loss for the shape
        shape_loss = []
        for i, a in enumerate(assignments):
            params = x[i, a[1], 3:]
            taylor = create_taylor_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
            pred = (params * taylor).sum(-1).T
            shape_loss.append(nn.MSELoss()(pred, y[i][1]))
        shape_loss = torch.stack(shape_loss).mean()
        # shape_loss = 0

        # loss for the segment prediction
        pred = x[..., 0]
        true = torch.zeros_like(pred).bool()
        for i, a in enumerate(assignments):
            true[i, a[1]] = True
        pred_loss = (nn.BCEWithLogitsLoss()(pred[true], torch.ones_like(pred[true])) +
                     nn.BCEWithLogitsLoss()(pred[~true], torch.zeros_like(pred[~true])))

        # loss for the position prediction
        pos_loss = []
        for i, a in enumerate(assignments):
            pos_loss.append(nn.L1Loss()(x[i, a[1], 1:3], y[i][0]))
        pos_loss = torch.stack(pos_loss).mean()

        return shape_loss, pred_loss, pos_loss

    def training_step(self, batch, batch_idx):
        simul, param, bifurcation = batch
        x = self(simul, param)
        shape_loss, pred_loss, pos_loss = self.loss(x, bifurcation)
        loss = 10 * shape_loss + pred_loss + pos_loss
        self.log("shape_loss", shape_loss, prog_bar=True)
        self.log("pred_loss", pred_loss, prog_bar=True)
        self.log("pos_loss", pos_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-4)


class BifurcationData(Dataset):
    def __init__(self, simul_data, par_data, bifurc_data):
        self.bifurc_data = []
        for bd in bifurc_data:
            bd = np.asarray(bd)
            if len(bd) > 0:
                pos = torch.asarray((bd[:, 0, 0], bd[:, -1, 0])).float()
                data = torch.asarray(bd[..., 1]).float()
                self.bifurc_data.append((pos.T, data))
            else:
                self.bifurc_data.append((torch.tensor([]), torch.tensor([])))
        self.simul_data = torch.asarray(simul_data).float()
        self.par_data = torch.asarray(par_data).float()

    def __len__(self):
        return len(self.bifurc_data)

    def __getitem__(self, idx):
        return self.simul_data[idx], self.par_data[idx], self.bifurc_data[idx]


def bifurcation_collate(batch):
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]), [b[2] for b in batch]


def create_taylor_vector(x0, x1, n=100):
    x = torch.asarray(np.linspace(x0.cpu(), x1.cpu(), n)).float().to(x0.device)
    # return np.stack([x, x**2/2, x**3/6, x**4/24, x**5/120, x**6/720, x**7/5040, x**8/40320, x**9/362880, x**10/3628800])
    return torch.stack([x ** 0, x ** 1, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9], dim=-1)


if __name__ == '__main__':
    equations = [('a*r*x - b*x**3', 'ab'), ()] * 1000
    bifurc_data = generate_data(equations, n_samples=100, abs_params='ab')
    data = []
    for i, eq in enumerate(equations):
        data.append(simulate(eq, n=100))
        if (i + 1) % 10 == 0: print('s', i + 1)
    simul_data, par_data = list(zip(*data))

    dataset = BifurcationData(simul_data, par_data, bifurc_data)
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=bifurcation_collate)
    pickle.dump(dataset, open('lightning_logs/version_3/dataset.pkl', 'wb'))

    model = BifurcationPredictor()
    trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', gradient_clip_val=0.1)
    trainer.fit(model, dataloader)
    print(equations)
