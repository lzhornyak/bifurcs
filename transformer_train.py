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


def simulate(equation, n=200, n_simul=400):
    f = eval('lambda x, t, r: ' + equation)
    x0 = np.random.rand(n) * 10 - 5
    t = np.geomspace(0.1, 10.1, n_simul) - 0.1
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

        self.n_features = 400
        pos_encoding = PositionalEncoding(self.n_features)
        encoder_layer = PosEncoderLayer(pos_encoding, d_model=self.n_features, nhead=10, dim_feedforward=1000, batch_first=True)
        self.encoder = PosEncoder(encoder_layer, num_layers=4)

        self.n_queries = 200
        self.queries = nn.Embedding(self.n_queries, 400)
        decoder_layer = PosDecoderLayer(pos_encoding, d_model=self.n_features, nhead=10, dim_feedforward=1000, batch_first=True)
        self.decoder = PosDecoder(decoder_layer, num_layers=4)

        self.final_decoder = nn.Linear(self.n_features, 13)

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
        for i in range(len(x)):
            if len(y[i][1]) > 0:
                params = x[i, :, 3:]
                # taylor = create_taylor_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
                # vector = create_bezier_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
                # pred = (params * vector.unsqueeze(-2)).sum(-1).T  # 20 x 100

                pred = x[i, :, 5:].unsqueeze(-2).expand(-1, y[i][1].shape[0], -1)
                targ = y[i][1][:, 1:-1].unsqueeze(0).expand(x.shape[1], -1, -1)
                cost_matrix = nn.L1Loss(reduction='none')(pred, targ).mean(dim=-1).T
                xp = x[i, :, 1:5].unsqueeze(0).expand(y[i][0].shape[0], -1, -1)
                yp = y[i][0].view(y[i][0].shape[0], -1).unsqueeze(1).expand(-1, x.shape[1], -1)
                cost_matrix += nn.L1Loss(reduction='none')(xp, yp).mean(dim=-1)

                assignments.append(linear_sum_assignment(cost_matrix.detach().cpu().numpy()))
                # cost_matrices.append(cost_matrix)
            else:
                assignments.append(([], []))

        # loss for the shape
        shape_loss = []
        for i, a in enumerate(assignments):
            params = x[i, a[1], 3:]
            if y[i][0].numel() == 0 and y[i][1].numel() == 0:
                shape_loss.append(torch.tensor(0, device=x.device))
                continue
            # taylor = create_taylor_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
            # vector = create_bezier_vector(y[i][0][:, 0], y[i][0][:, 1], n=y[i][1].shape[1])
            shape_loss.append(nn.L1Loss()(x[i, a[1], 5:], y[i][1][:, 1:-1]))
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
            if y[i][0].numel() == 0:
                pos_loss.append(torch.tensor(0, device=x.device))
                continue
            pos_loss.append(nn.L1Loss()(x[i, a[1], 1:5], y[i][0].view(y[i][0].shape[0], -1)))
        pos_loss = torch.stack(pos_loss).mean()

        return shape_loss, pred_loss, pos_loss

    def training_step(self, batch, batch_idx):
        simul, param, bifurcation = batch

        # random trajectory dropout
        match np.random.rand():
            case x if x > 0.75:
                mask = np.random.rand(simul.shape[1]) > np.random.rand()
                simul, param = simul[:, mask], param[:, mask]
            case x if 0.75 > x > 0.5:
                bounds = sorted(np.random.rand(2))
                mask = np.logical_and(param[0, :] > bounds[0], param[0, :] < bounds[1])
                simul, param = simul[:, mask], param[:, mask]
            case x if 0.5 > x > 0.25:
                bounds = sorted(np.random.rand(2))
                mask = np.logical_and(simul[-1, :] > bounds[0], simul[-1, :] < bounds[1])
                simul, param = simul[:, mask], param[:, mask]

        x = self(simul, param)
        shape_loss, pred_loss, pos_loss = self.loss(x, bifurcation)
        loss = 50 * shape_loss + pred_loss / 2 + 200 * pos_loss
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
                bounds = torch.asarray(np.stack(bd[..., 0])).float()
                data = torch.asarray(np.stack(bd[..., 1])).float()
                self.bifurc_data.append((bounds, data))
            else:
                self.bifurc_data.append((torch.tensor([]), torch.tensor([])))
        self.simul_data = torch.asarray(simul_data).float()
        self.par_data = torch.asarray(par_data).float()

    def __len__(self):
        return len(self.bifurc_data)

    def __getitem__(self, idx):
        par_datum = self.par_data[idx]
        sort_mask = np.argsort(par_datum)
        return self.simul_data[idx][sort_mask], self.par_data[idx][sort_mask], self.bifurc_data[idx]


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
        equations, dataset = pickle.load(open('dataset.pkl', 'rb'))
    except FileNotFoundError:
        equations = generate_equations(['1', 'x**1', 'x**2', 'x**3'], append='-x**5') * 1000
        bifurc_data = generate_data(equations, n_samples=250, abs_params='')
        data = []
        for i, eq in enumerate(equations):
            data.append(simulate(eq, n=200, n_simul=400))
            if (i + 1) % 10 == 0: print('s', i + 1)
        simul_data, par_data = list(zip(*data))
        pickle.dump((equations, simul_data, par_data, bifurc_data), open('data.pkl', 'wb'))
        data = (equations, simul_data, par_data, bifurc_data)
        dataset = BifurcationData(*data[1:])

    dataloader = DataLoader(dataset, batch_size=20, collate_fn=bifurcation_collate)

    model = BifurcationPredictor()
    trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', gradient_clip_val=0.1)
    trainer.fit(model, dataloader)
