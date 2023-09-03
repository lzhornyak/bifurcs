import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from generate_bifurcs import *
from scipy.integrate import odeint
from scipy.optimize import linear_sum_assignment
import pytorch_lightning as pl


def simulate(equation, n=100):
    f = eval('lambda x, t, r: ' + equation)
    x0 = np.random.rand(n) * 20 - 10
    t = np.geomspace(0.1, 10.1, 100) - 0.1
    r = np.random.rand(n) * 4 - 2
    x = odeint(f, x0, t, args=(r,))
    return x.T, r


class BifurcationPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=5, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.transformer = nn.Transformer(d_model=100, nhead=5, num_encoder_layers=3, num_decoder_layers=3,
                                          dim_feedforward=100, dropout=0.1, batch_first=True)
        self.decoder = nn.Linear(100, 100)
        # self.emplacer = nn.Linear(1, 100)
        # self.pos_encoding = PositionalEncoding(100)
        self.embedding = nn.Embedding(20, 100)

    def forward(self, x, r):
        # x = self.pos_encoding(x, r)
        # x = self.emplacer(r.unsqueeze(-1)) + x
        x[..., -1] = r
        embedding = self.embedding(torch.arange(20, device=x.device)).expand(x.shape[0], -1, -1)
        x = self.transformer(x, embedding)
        return self.decoder(x)

    def loss(self, x, y):
        # x : (batch, 100, 100)
        # y : (batch, (<positions>, segments, 97))
        # loss for the shape

        # determine assignments
        assignments = []
        cost_matrices = []
        for i in range(len(x)):
            if len(y[i][1]) > 0:
                xp = x[i, :, 1:3].unsqueeze(0).expand(y[i][0].shape[0], -1, -1)
                yp = y[i][0].unsqueeze(1).expand(-1, x.shape[1], -1)
                cost_matrix = nn.L1Loss(reduction='none')(xp, yp).mean(dim=-1)
                assignments.append(linear_sum_assignment(cost_matrix.detach().cpu().numpy()))
            else:
                assignments.append(([], []))

        # loss for the shape
        shape_loss = []
        for i, a in enumerate(assignments):
            if len(a[0]) > 0:
                shape_loss.append(nn.MSELoss()(x[i, a[1], 3:], y[i][1]))
        shape_loss = torch.stack(shape_loss).mean()

        # loss for the segment prediction
        pred = x[..., 0]
        true = torch.zeros_like(pred)
        for i, a in enumerate(assignments):
            true[i, a[1]] = 1
        pred_loss = nn.BCEWithLogitsLoss()(pred, true)

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
        loss = shape_loss + pred_loss + pos_loss
        self.log("shape_loss", shape_loss, prog_bar=True)
        self.log("pred_loss", pred_loss, prog_bar=True)
        self.log("pos_loss", pos_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# class PositionalEncoding(nn.Module):
#     # modified from pytorch.org tutorials
#     def __init__(self, d_model: int, dropout: float = 0.1):
#         super().__init__()
#         # self.dropout = nn.Dropout(p=dropout)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(4.) / d_model))
#         self.register_buffer('div_term', div_term.cuda())
#
#     def forward(self, x, r):
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         pe = torch.zeros_like(x)
#         pe[..., 0::2] = torch.sin(r[..., None] * self.div_term)
#         pe[..., 1::2] = torch.cos(r[..., None] * self.div_term)
#         # return self.dropout(x)
#         return x + pe


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


if __name__ == '__main__':
    equations = [('a*r*x - b*x**2 - 0.01*c*x**5', 'abc')] * 10
    bifurc_data = generate_data(equations, n_samples=97)
    simul_data, par_data = list(zip(*[simulate(eq) for eq in equations]))

    dataset = BifurcationData(simul_data, par_data, bifurc_data)
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=bifurcation_collate)

    model = BifurcationPredictor()
    trainer = pl.Trainer(max_epochs=1000)
    trainer.fit(model, dataloader)
    print(equations)
