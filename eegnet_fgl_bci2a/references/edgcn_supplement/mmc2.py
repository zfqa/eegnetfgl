import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class ClassificationModule(nn.Module):
    def __init__(self, model, n_classes, **kwargs):
        super().__init__()
        self.model = model
        self.n_classes = n_classes

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# --- [Section 3.1] Local Feature Extraction ---
class LocalFeatureExtraction(nn.Module):
    def __init__(self, in_channels, ghost_channels=16):
        super(LocalFeatureExtraction, self).__init__()
        # Ghost Module: Primary Conv
        self.primary_conv = nn.Conv2d(1, ghost_channels, kernel_size=(1, 1))
        # Ghost Module: Cheap Operation
        self.cheap_operation = nn.Conv2d(ghost_channels, 1, kernel_size=(3, 1), padding=(1, 0), groups=1)

        # Parallel Temporal Convolutions (1x40, 1x60, 1x80)
        self.temporal_conv_40 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 40), padding=(0, 0)),
            nn.BatchNorm2d(10),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 5)),
            nn.Dropout(p=0.5)
        )
        self.temporal_conv_60 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 60), padding=(0, 10)),
            nn.BatchNorm2d(10),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 5)),
            nn.Dropout(p=0.5)
        )
        self.temporal_conv_80 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=(1, 80), padding=(0, 20)),
            nn.BatchNorm2d(10),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 5)),
            nn.Dropout(p=0.5)
        )
        # Fusion Convolution (1x15)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(30, 30, kernel_size=(1, 15)),
            nn.BatchNorm2d(30),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 6)),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):

        x = x.unsqueeze(1)

        # Ghost feature generation
        x = self.primary_conv(x)
        x = self.cheap_operation(x)

        f1 = self.temporal_conv_40(x)
        f2 = self.temporal_conv_60(x)
        f3 = self.temporal_conv_80(x)

        # Align time dimensions
        min_time = min(f1.shape[-1], f2.shape[-1], f3.shape[-1])
        f1 = f1[..., :min_time]
        f2 = f2[..., :min_time]
        f3 = f3[..., :min_time]

        # Feature Concatenation and Fusion
        x_cat = torch.cat([f1, f2, f3], dim=1)
        fused = self.fusion_conv(x_cat)

        # Rearrange for GCN: (N, C, T, F)
        fused = fused.permute(0, 2, 3, 1).contiguous()
        return fused

class SASE(nn.Module):
    def __init__(self, num_channels, embed_dim):
        super(SASE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, local_graph, global_mask):
        # Fusion of local and global structures
        local_avg = local_graph.mean(dim=0)
        global_avg = global_mask.mean(dim=0)

        # Feature extraction from graph structures
        node_feature = torch.cat([local_avg, global_avg], dim=-1)

        # Generate spatial embedding E_s
        E_s = self.mlp(node_feature)

        # Construct fusion graph for GCN support
        fusion_graph = 0.5 * local_avg + 0.5 * global_avg
        return E_s, fusion_graph

class MRTE(nn.Module):
    def __init__(self, input_dim=44, hidden_dim=32):
        super(MRTE, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

    def forward(self, psd_interp, psd_down):
        # Pooling over windows
        pooled_interp = psd_interp.mean(dim=1)
        pooled_down = psd_down.mean(dim=1)

        # Fusion of multi-resolution features
        feat = (pooled_interp + pooled_down) / 2

        # Generate temporal embedding E_t
        h_last = self.mlp(feat)
        return h_last


class EmbeddingDrivenGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, time_len, meta_axis="S"):
        super(EmbeddingDrivenGCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cheb_k = cheb_k
        self.embed_dim = embed_dim
        self.meta_axis = meta_axis.upper()  #

        # Shared Parameter Bank (Weights & Bias)
        self.weights_pool = nn.Parameter(torch.randn(embed_dim, cheb_k * time_len * input_dim, output_dim))
        self.bias_pool = nn.Parameter(torch.randn(embed_dim, output_dim))
        nn.init.xavier_uniform_(self.weights_pool)

    def forward(self, x, support, embedding):
        B, N, T, F_T = x.shape
        x = x.reshape(B, N, T * F_T)
        x_g = []


        if support.dim() == 2:
            g_list = [torch.eye(N, device=x.device), support]
            for k in range(2, self.cheb_k):
                g_list.append(2 * support @ g_list[-1] - g_list[-2])
            for g in g_list:
                x_g.append(torch.einsum("nm,bmc->bnc", g, x))
        elif support.dim() == 3:
            g_list = [torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1), support]
            for k in range(2, self.cheb_k):
                g_list.append(2 * support @ g_list[-1] - g_list[-2])
            for g in g_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", g, x))

        x_cat = torch.cat(x_g, dim=-1)


        if self.meta_axis == "S":
            # Spatial Embedding drives parameters per node
            weights = torch.einsum("nd,dio->nio", embedding, self.weights_pool)
            bias = embedding @ self.bias_pool
            out = torch.einsum("bni,nio->bno", x_cat, weights) + bias.unsqueeze(0)
        elif self.meta_axis == "T":
            # Temporal Embedding drives parameters per sample
            weights = torch.einsum("bd,dio->bio", embedding, self.weights_pool)
            bias = embedding @ self.bias_pool
            out = torch.einsum("bni,bio->bno", x_cat, weights) + bias.unsqueeze(1)

        return out

class EDGCNModule(nn.Module):
    def __init__(
            self,
            in_channels: int = 22,
            n_classes: int = 4,

            d_model: int = 32,
            cheb_k: int = 3,
            time_len: int = 33,
            ghost_channels: int = 11,
            embed_input_dim: int = 88
    ):
        super(EDGCNModule, self).__init__()

        self.n_classes = n_classes


        self.local_feature_extractor = LocalFeatureExtraction(in_channels=in_channels, ghost_channels=ghost_channels)

        self.sase = SASE(num_channels=in_channels, embed_dim=d_model)
        self.temporal_embedder = MRTE(input_dim=embed_input_dim, hidden_dim=d_model)

        self.edgcn_t = EmbeddingDrivenGCN(input_dim=30, output_dim=d_model, cheb_k=cheb_k, embed_dim=d_model,
                                          time_len=time_len, meta_axis="T")
        self.edgcn_s = EmbeddingDrivenGCN(input_dim=30, output_dim=d_model, cheb_k=cheb_k, embed_dim=d_model,
                                          time_len=time_len, meta_axis="S")

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, x1, x2, edge, local_graph_weight, global_mask):

        x_encoder = self.local_feature_extractor(x)  # -> (B, C, T', F)

        Q_s, fusion_graph = self.sase(local_graph_weight, global_mask)  # Spatial Embedding
        Q_t = self.temporal_embedder(x1, x2)  # Temporal Embedding

        x_t = self.edgcn_t(x_encoder, fusion_graph, Q_t)
        x_s = self.edgcn_s(x_encoder, fusion_graph, Q_s)

        adaptive_stream = (x_s + x_t)

        # Classification
        logits = self.classifier(adaptive_stream)  # (B, N, n_classes)
        logits = logits.mean(dim=1)  # Global Average Pooling over nodes


        return logits

    @property
    def theta_pool(self):
        return [self.edgcn_t.weights_pool, self.edgcn_s.weights_pool]

class EDGCN(ClassificationModule):
    def __init__(
            self,
            in_channels: int = 22,
            n_classes: int = 4,
            **kwargs
    ):

        model = EDGCNModule(
            in_channels=in_channels,
            n_classes=n_classes,
            d_model=32,
            cheb_k=3,
            ghost_channels=11
        )
        super(EDGCN, self).__init__(model, n_classes, **kwargs)