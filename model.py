import torch
import torch.nn.functional as F
from torch import nn, tensor, Tensor
from ctypes import Union
from typing import Tuple


class TC(nn.Module):
    def __init__(
            self,
            d_model,
            nhead=3,
            num_layers=6,
            dim_feedforward=2048,
            hidden_size = 512,
            out_features = 2,
            dropout=0.1,):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be dividable by num_heads!"

        self.emb = Embedding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            # nn.Linear(d_model, hidden_size),
            # nn.Dropout(p=0.5),
            nn.Linear(d_model, out_features),
            nn.Dropout(p=0.2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, -1, :])
        return x


class Embedding(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="sum"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def _forward(self, logits, label):
        '''
        :param logits: model output results, pls do sigmoid by yourself kindly
        :param label: tagets labels as usual
        :return: focal loss
        '''
        coeff = torch.abs(label - logits).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1 , 50),
                                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 50),
                                  -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1.0 - label) * (1.0 - self.alpha) * log_1_probs
        loss *= coeff

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        return loss

    def forward(self, logits, label):
        p_1 = - self.alpha * torch.pow(1 - logits, self.gamma) * torch.log2(logits) * label
        p_0 = - (1 - self.alpha) * torch.pow(logits, self.gamma) * torch.log2(1 - logits) * (1 - label)
        return (p_0 + p_1).sum()


if __name__ == '__main__':
    in_features = 2
    hidden_size = 4
    num_layers = 2
    batch_size = 1
    out_features = 1

    tgt = tensor([.0, .1])
    x = torch.tensor([
        [
            [0.5293, 0.0178, 0.0000, 0.3692, 0.4225, 0.4448, 0.0178, 0.4448, 0.0934,
             0.0000, 0.0000, 0.0000],
            [0.5061, 0.0181, 0.0000, 0.3751, 0.4293, 0.4519, 0.0181, 0.4519, 0.0994,
             0.0000, 0.0000, 0.0000],
            [0.5256, 0.0178, 0.0000, 0.3697, 0.4232, 0.4454, 0.0178, 0.4454, 0.1025,
             0.0000, 0.0000, 0.0000],
            [0.5293, 0.0178, 0.0000, 0.3692, 0.4225, 0.4448, 0.0178, 0.4448, 0.0934,
             0.0000, 0.0000, 0.0000],
            [0.5325, 0.0177, 0.0000, 0.3683, 0.4215, 0.4437, 0.0177, 0.4437, 0.0932,
             0.0000, 0.0000, 0.0000],
            [0.5325, 0.0177, 0.0000, 0.3683, 0.4215, 0.4437, 0.0177, 0.4437, 0.0932,
             0.0000, 0.0000, 0.0000],
            [0.5263, 0.0178, 0.0000, 0.3702, 0.4237, 0.4460, 0.0178, 0.4460, 0.0892,
             0.0000, 0.0000, 0.0000],
            [0.5194, 0.0179, 0.0000, 0.3716, 0.4253, 0.4477, 0.0179, 0.4477, 0.0985,
             0.0000, 0.0000, 0.0000],
            [0.5320, 0.0177, 0.0000, 0.3680, 0.4212, 0.4434, 0.0177, 0.4434, 0.1020,
             0.0000, 0.0000, 0.0000],
            [0.5256, 0.0178, 0.0000, 0.3697, 0.4232, 0.4454, 0.0178, 0.4454, 0.1025,
             0.0000, 0.0000, 0.0000],
            [0.5288, 0.0178, 0.0000, 0.3688, 0.4222, 0.4444, 0.0178, 0.4444, 0.1022,
             0.0000, 0.0000, 0.0000],
            [0.5288, 0.0178, 0.0000, 0.3688, 0.4222, 0.4444, 0.0178, 0.4444, 0.1022,
             0.0000, 0.0000, 0.0000],
            [0.5191, 0.0179, 0.0000, 0.3714, 0.4251, 0.4475, 0.0179, 0.4475, 0.1029,
             0.0000, 0.0000, 0.0000],
            [0.5025, 0.0181, 0.0000, 0.3757, 0.4301, 0.4527, 0.0181, 0.4527, 0.1041,
             0.0000, 0.0000, 0.0000],
            [0.5061, 0.0181, 0.0000, 0.3751, 0.4293, 0.4519, 0.0181, 0.4519, 0.0994,
             0.0000, 0.0000, 0.0000],
            [0.5291, 0.0178, 0.0000, 0.3690, 0.4224, 0.4446, 0.0178, 0.4446, 0.0978,
             0.0000, 0.0000, 0.0000],
            [0.5094, 0.0180, 0.0000, 0.3742, 0.4283, 0.4508, 0.0180, 0.4508, 0.0992,
             0.0000, 0.0000, 0.0000],
            [0.5194, 0.0179, 0.0000, 0.3716, 0.4253, 0.4477, 0.0179, 0.4477, 0.0985,
             0.0000, 0.0000, 0.0000],
            [0.5194, 0.0179, 0.0000, 0.3716, 0.4253, 0.4477, 0.0179, 0.4477, 0.0985,
             0.0000, 0.0000, 0.0000],
            [0.5231, 0.0179, 0.0000, 0.3711, 0.4247, 0.4471, 0.0179, 0.4471, 0.0894,
             0.0000, 0.0000, 0.0000],
            [0.5228, 0.0179, 0.0000, 0.3709, 0.4245, 0.4469, 0.0179, 0.4469, 0.0938,
             0.0000, 0.0000, 0.0000],
            [0.5161, 0.0180, 0.0000, 0.3725, 0.4263, 0.4488, 0.0180, 0.4488, 0.0987,
             0.0000, 0.0000, 0.0000],
            [0.5259, 0.0178, 0.0000, 0.3699, 0.4234, 0.4456, 0.0178, 0.4456, 0.0980,
             0.0000, 0.0000, 0.0000],
            [0.5259, 0.0178, 0.0000, 0.3699, 0.4234, 0.4456, 0.0178, 0.4456, 0.0980,
             0.0000, 0.0000, 0.0000],
            [0.4749, 0.0184, 0.0000, 0.3827, 0.4380, 0.4611, 0.0184, 0.4611, 0.1014,
             0.0000, 0.0000, 0.0000],
            [0.5128, 0.0180, 0.0000, 0.3733, 0.4273, 0.4498, 0.0180, 0.4498, 0.0990,
             0.0000, 0.0000, 0.0000],
            [0.5215, 0.0178, 0.0000, 0.3744, 0.4235, 0.4457, 0.0178, 0.4457, 0.1025,
             0.0000, 0.0000, 0.0000],
            [0.5114, 0.0179, 0.0000, 0.3768, 0.4262, 0.4486, 0.0179, 0.4486, 0.1077,
             0.0000, 0.0000, 0.0000],
            [0.5217, 0.0178, 0.0000, 0.3746, 0.4236, 0.4459, 0.0178, 0.4459, 0.0981,
             0.0000, 0.0000, 0.0000],
            [0.5248, 0.0178, 0.0000, 0.3736, 0.4225, 0.4447, 0.0178, 0.4447, 0.1023,
             0.0000, 0.0000, 0.0000],
            [0.5250, 0.0178, 0.0000, 0.3737, 0.4227, 0.4449, 0.0178, 0.4449, 0.0979,
             0.0000, 0.0000, 0.0000],
            [0.5162, 0.0180, 0.0000, 0.3770, 0.4219, 0.4489, 0.0180, 0.4489, 0.0988,
             0.0000, 0.0000, 0.0000],
            [0.5197, 0.0179, 0.0000, 0.3763, 0.4211, 0.4480, 0.0179, 0.4480, 0.0941,
             0.0000, 0.0000, 0.0000],
            [0.5160, 0.0179, 0.0000, 0.3769, 0.4217, 0.4487, 0.0179, 0.4487, 0.1032,
             0.0000, 0.0000, 0.0000],
            [0.4926, 0.0182, 0.0000, 0.3831, 0.4287, 0.4561, 0.0182, 0.4561, 0.1003,
             0.0000, 0.0000, 0.0000],
            [0.4999, 0.0182, 0.0000, 0.3817, 0.4272, 0.4544, 0.0182, 0.4544, 0.0909,
             0.0000, 0.0000, 0.0000],
            [0.5262, 0.0178, 0.0000, 0.3746, 0.4192, 0.4459, 0.0178, 0.4459, 0.0936,
             0.0000, 0.0000, 0.0000],
            [0.5230, 0.0179, 0.0000, 0.3755, 0.4202, 0.4470, 0.0179, 0.4470, 0.0939,
             0.0000, 0.0000, 0.0000],
            [0.4928, 0.0183, 0.0000, 0.3833, 0.4289, 0.4563, 0.0183, 0.4563, 0.0958,
             0.0000, 0.0000, 0.0000],
            [0.5230, 0.0179, 0.0000, 0.3755, 0.4202, 0.4470, 0.0179, 0.4470, 0.0939,
             0.0000, 0.0000, 0.0000],
            [0.5131, 0.0180, 0.0000, 0.3781, 0.4231, 0.4501, 0.0180, 0.4501, 0.0945,
             0.0000, 0.0000, 0.0000],
            [0.5164, 0.0180, 0.0000, 0.3772, 0.4221, 0.4491, 0.0180, 0.4491, 0.0943,
             0.0000, 0.0000, 0.0000],
            [0.5133, 0.0180, 0.0000, 0.3782, 0.4233, 0.4503, 0.0180, 0.4503, 0.0901,
             0.0000, 0.0000, 0.0000],
            [0.5294, 0.0178, 0.0000, 0.3737, 0.4182, 0.4449, 0.0178, 0.4449, 0.0934,
             0.0000, 0.0000, 0.0000],
            [0.5199, 0.0179, 0.0000, 0.3765, 0.4213, 0.4482, 0.0179, 0.4482, 0.0896,
             0.0000, 0.0000, 0.0000],
            [0.4999, 0.0182, 0.0000, 0.3817, 0.4272, 0.4544, 0.0182, 0.4544, 0.0909,
             0.0000, 0.0000, 0.0000],
            [0.5230, 0.0179, 0.0000, 0.3755, 0.4202, 0.4470, 0.0179, 0.4470, 0.0939,
             0.0000, 0.0000, 0.0000],
            [0.5294, 0.0178, 0.0000, 0.3737, 0.4182, 0.4449, 0.0178, 0.4449, 0.0934,
             0.0000, 0.0000, 0.0000],
            [0.5232, 0.0179, 0.0000, 0.3756, 0.4203, 0.4472, 0.0179, 0.4472, 0.0894,
             0.0000, 0.0000, 0.0000],
            [0.5133, 0.0180, 0.0000, 0.3782, 0.4233, 0.4503, 0.0180, 0.4503, 0.0901,
             0.0000, 0.0000, 0.0000]],
    ])

    tc = TC(d_model=12, out_features=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.to(device)
    tc.to(device)
    out = tc(x)
    print(out)
    print(x.shape)
    print(out.shape)

    fl = FocalLoss()
    fl = fl.to(device)
    print(fl(out, tgt.to(device)))
    print(fl._forward(out, tgt.to(device)))