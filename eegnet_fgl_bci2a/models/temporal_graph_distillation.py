"""Paper-guided 4s teachers and a fixed-2s Compact relational student.

The classifiers and cross-architecture relation objective are new adaptations,
not a reproduction of EDGCN's reported classification scores.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .edgcn import EDGCNStudent
from .eegnet import EEGNet


class MultiscaleTemporalTeacher(nn.Module):
    """Ghost / 20,60,80 temporal kernels / crop / pool / fusion / local head."""

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.primary = nn.Conv2d(1, 10, (1, 5), padding=(0, 2), bias=False)
        self.cheap = nn.Conv2d(10, 10, (1, 3), padding=(0, 1), groups=10, bias=False)
        self.branches = nn.ModuleList(nn.Conv2d(20, 10, (1, k), bias=False) for k in (20, 60, 80))
        self.pool = nn.MaxPool2d((1, 2))
        self.fusion = nn.Sequential(nn.Conv2d(30, 30, (1, 15), bias=False),
                                    nn.BatchNorm2d(30), nn.ELU(), nn.MaxPool2d((1, 4)),
                                    nn.Dropout(.5))
        # Keep electrode identity; do not average left/right electrodes together.
        self.readout = nn.AdaptiveAvgPool2d((n_channels, 4))
        self.classifier = nn.Linear(30 * n_channels * 4, n_classes)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, self.n_channels, 1000):
            raise ValueError("Temporal teacher requires [B,1,C,1000] (4s at250Hz)")
        primary = self.primary(x)
        ghost = torch.cat((primary, self.cheap(primary)), 1)
        paths = [branch(ghost) for branch in self.branches]
        length = min(p.shape[-1] for p in paths)
        fused = self.fusion(torch.cat([self.pool(p[..., :length]) for p in paths], 1))
        return self.classifier(self.readout(fused).flatten(1))


class EDGCNGraphTeacher(EDGCNStudent):
    """Train full CE classifier; expose pre-readout spatial graph nodes for KD."""

    window_seconds = 4

    def forward_with_nodes(self, x):
        temporal, spatial = self.graph_features(x)
        logits = self.classifier(F.elu(torch.cat((temporal.mean(1), spatial.mean(1)), -1)))
        return logits, spatial


class CompactRelationStudent(EEGNet):
    """Unchanged Compact classifier; no extra deployment parameters or graph.

    Node features: per-electrode, per-temporal-filter centered log-power in four
    consecutive bins. This relation loss reaches temporal conv/BN, not the
    subsequent depthwise spatial-convolution weights directly.
    """

    def __init__(self, n_channels, n_classes):
        super().__init__(n_channels, 500, n_classes)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, self.n_channels, 500):
            raise ValueError("Student requires [B,1,C,500]; crop before calling")
        return super().forward(x)

    def forward_with_nodes(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, self.n_channels, 500):
            raise ValueError("Student requires [B,1,C,500]; crop before calling")
        h = self.features[1](self.features[0](x))
        # Compact's even-kernel padding produces501 positions; exclude the extra.
        chunks = h[..., :500].reshape(len(h), h.shape[1], self.n_channels, 4, 125)
        chunks = chunks - chunks.mean(-1, keepdim=True)
        nodes = (chunks.square().mean(-1) + 1e-6).log().permute(0, 2, 1, 3).flatten(2)
        for layer in self.features[2:]:
            h = layer(h)
        return self.classifier(h.flatten(1)), nodes


def node_relations(nodes):
    if nodes.ndim != 3 or nodes.shape[1] < 2:
        raise ValueError("Expected node features [B,C,D] with C>=2")
    normalized = F.normalize(nodes, dim=-1, eps=1e-6)
    return normalized @ normalized.transpose(-1, -2)


def relation_loss(student_nodes, teacher_relations):
    student = node_relations(student_nodes)
    target = teacher_relations.detach().to(student)
    if target.shape != student.shape:
        raise ValueError("Teacher/student electrode order and count must match")
    mask = ~torch.eye(student.shape[1], device=student.device, dtype=torch.bool)
    return (student[:, mask] - target[:, mask]).square().mean()


def temporal_kd(student_logits, teacher_logits, temperature=1.):
    if temperature <= 0:
        raise ValueError("KD temperature must be positive")
    target = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return F.kl_div(F.log_softmax(student_logits / temperature, dim=-1), target,
                    reduction="batchmean") * temperature ** 2
