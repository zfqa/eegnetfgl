"""Core from user-supplied EDGCN mmc2.py; wrapper/interface cleanup only.

See references/edgcn_supplement for unchanged source files. This follows CODE
(40/60/80, sum, node mean), not the conflicting figure/text. Auxiliary PSD and
graph generation are external in the supplement and must be documented apart.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LocalFeatureExtraction(nn.Module):
    def __init__(self, in_channels, ghost_channels=16):
        super().__init__()
        self.primary_conv = nn.Conv2d(1, ghost_channels, kernel_size=(1, 1))
        self.cheap_operation = nn.Conv2d(ghost_channels, 1, (3, 1), padding=(1, 0), groups=1)
        for width, padding in ((40, 0), (60, 10), (80, 20)):
            setattr(self, f"temporal_conv_{width}", nn.Sequential(
                nn.Conv2d(1, 10, (1, width), padding=(0, padding)), nn.BatchNorm2d(10),
                nn.ELU(), nn.MaxPool2d((1, 5)), nn.Dropout(.5)))
        self.fusion_conv = nn.Sequential(nn.Conv2d(30, 30, (1, 15)), nn.BatchNorm2d(30),
                                         nn.ELU(), nn.MaxPool2d((1, 6)), nn.Dropout(.5))

    def forward(self, x):
        x = self.cheap_operation(self.primary_conv(x.unsqueeze(1)))
        paths = [self.temporal_conv_40(x), self.temporal_conv_60(x), self.temporal_conv_80(x)]
        length = min(v.shape[-1] for v in paths)
        fused = self.fusion_conv(torch.cat([v[..., :length] for v in paths], 1))
        return fused.permute(0, 2, 3, 1).contiguous()


class SASE(nn.Module):
    def __init__(self, num_channels, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_channels * 2, 64), nn.ReLU(), nn.Linear(64, embed_dim))

    def forward(self, local_graph, global_mask):
        local_avg, global_avg = local_graph.mean(0), global_mask.mean(0)
        return self.mlp(torch.cat((local_avg, global_avg), -1)), .5 * local_avg + .5 * global_avg


class MRTE(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, hidden_dim))

    def forward(self, psd_interp, psd_down):
        return self.mlp((psd_interp.mean(1) + psd_down.mean(1)) / 2)


class EmbeddingDrivenGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, time_len, meta_axis="S"):
        super().__init__()
        if cheb_k < 2:
            raise ValueError("Supplement constructs at least two supports; this port requires K>=2")
        self.input_dim, self.output_dim = input_dim, output_dim
        self.cheb_k, self.embed_dim, self.meta_axis = cheb_k, embed_dim, meta_axis.upper()
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k * time_len * input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_pool)

    def forward(self, x, support, embedding):
        batch, nodes, time, features = x.shape
        x = x.reshape(batch, nodes, time * features)
        if support.dim() == 2:
            terms = [torch.eye(nodes, device=x.device), support]
            for _ in range(2, self.cheb_k):
                terms.append(2 * support @ terms[-1] - terms[-2])
            propagated = [torch.einsum("nm,bmc->bnc", g, x) for g in terms]
        elif support.dim() == 3:
            terms = [torch.eye(nodes, device=x.device).unsqueeze(0).repeat(batch, 1, 1), support]
            for _ in range(2, self.cheb_k):
                terms.append(2 * support @ terms[-1] - terms[-2])
            propagated = [torch.einsum("bnm,bmc->bnc", g, x) for g in terms]
        else:
            raise ValueError("Support must be2D or3D")
        concatenated = torch.cat(propagated, -1)
        # The supplement uses the supplied fusion adjacency directly, not -DAD.
        if self.meta_axis == "S":
            weights = torch.einsum("nd,dio->nio", embedding, self.weights_pool)
            return torch.einsum("bni,nio->bno", concatenated, weights) + (embedding @ self.bias_pool).unsqueeze(0)
        if self.meta_axis == "T":
            weights = torch.einsum("bd,dio->bio", embedding, self.weights_pool)
            return torch.einsum("bni,bio->bno", concatenated, weights) + (embedding @ self.bias_pool).unsqueeze(1)
        raise ValueError("Unknown meta axis")


class EDGCNModule(nn.Module):
    def __init__(self, in_channels=22, n_classes=4, d_model=32, cheb_k=3,
                 time_len=33, ghost_channels=11, embed_input_dim=88):
        super().__init__()
        self.n_classes = n_classes
        self.local_feature_extractor = LocalFeatureExtraction(in_channels, ghost_channels)
        self.sase = SASE(in_channels, d_model)
        self.temporal_embedder = MRTE(embed_input_dim, d_model)
        self.edgcn_t = EmbeddingDrivenGCN(30, d_model, cheb_k, d_model, time_len, "T")
        self.edgcn_s = EmbeddingDrivenGCN(30, d_model, cheb_k, d_model, time_len, "S")
        self.classifier = nn.Linear(d_model, n_classes)

    def forward_with_nodes(self, x, x1, x2, edge, local_graph_weight, global_mask):
        encoded = self.local_feature_extractor(x)
        spatial, support = self.sase(local_graph_weight, global_mask)
        temporal = self.temporal_embedder(x1, x2)
        xt = self.edgcn_t(encoded, support, temporal)
        xs = self.edgcn_s(encoded, support, spatial)
        return self.classifier(xs + xt).mean(1), xs

    def forward(self, x, x1, x2, edge, local_graph_weight, global_mask):
        return self.forward_with_nodes(x, x1, x2, edge, local_graph_weight, global_mask)[0]

    @property
    def theta_pool(self):
        return [self.edgcn_t.weights_pool, self.edgcn_s.weights_pool]


def orthogonal_loss(pool):
    # mmc1.py uses the WHOLE bank, not separate Chebyshev-order sub-blocks.
    dim = pool.shape[0]
    flat = pool.view(dim, -1)
    sim = flat @ flat.T
    norm = torch.norm(flat, dim=1, keepdim=True) + 1e-8
    cosine = sim / (norm @ norm.T)
    return ((cosine - torch.eye(dim, device=pool.device)) ** 2).sum() / dim ** 2


def objective(model, logits, labels):
    # mmc1 training_step: fixed lambda=.1 times the two whole-bank penalties.
    return F.cross_entropy(logits, labels) + .1 * sum(orthogonal_loss(p) for p in model.theta_pool)
