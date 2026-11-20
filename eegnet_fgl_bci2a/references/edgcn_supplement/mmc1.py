import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from channel_attention.test.test2 import output
from channel_attention.utils.interaug import interaug
from channel_attention.utils.lr_scheduler import linear_warmup_cosine_decay
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

def orthogonal_loss(P):

    D, M, N = P.shape
    P_flat = P.view(D, -1)  # (D, M*N)
    sim = torch.matmul(P_flat, P_flat.T)  # (D, D)
    norm = torch.norm(P_flat, dim=1, keepdim=True) + 1e-8
    norm_matrix = norm @ norm.T  # (D, D)

    cos_sim = sim / norm_matrix  # cosine similarity matrix
    identity = torch.eye(D, device=P.device)

    loss = ((cos_sim - identity) ** 2).sum() / (D ** 2)
    return loss

class ClassificationModule(pl.LightningModule):
    def __init__(
            self,
            model,
            n_classes,
            lr=0.001,
            weight_decay=0.0,
            optimizer="adam",
            scheduler=False,
            max_epochs=1000,
            warmup_epochs=20,

            **kwargs
    ):
        super(ClassificationModule, self).__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.validation_preds = []
        self.validation_labels = []
        self.test_preds = []
        self.test_labels = []
        self.test_round = 0
    def forward(self, x,x_merged,x_downsampled,edges,local_graph_weight,global_mask):
        output= self.model(x,x_merged,x_downsampled,edges,local_graph_weight,global_mask)

        return output

    def configure_optimizers(self):
        betas = self.hparams.get("beta_1", 0.9), self.hparams.get("beta_2", 0.999)
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                         betas=betas,
                                         weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "adamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                          betas=betas,
                                          weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
        else:
            raise NotImplementedError
        if self.hparams.scheduler:
            scheduler = LambdaLR(optimizer,
                                 linear_warmup_cosine_decay(self.hparams.warmup_epochs,
                                                            self.hparams.max_epochs))
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        loss, acc, _, _ = self.shared_step(batch, batch_idx, mode="train", alpha=0.5)

        if hasattr(self.model, "theta_pool"):
            pools = self.model.theta_pool
            loss_orth_total = 0
            for P in pools:
                loss_orth_total += orthogonal_loss(P)

            λ = 0.1
            loss = loss + λ * loss_orth_total

            self.log("orth_loss", loss_orth_total, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, preds, labels = self.shared_step(batch, batch_idx, mode="val")
        self.validation_preds.append(preds)
        self.validation_labels.append(labels)
        return {"val_loss": loss, "val_acc": acc,"preds": preds, "labels": labels}

    def on_validation_epoch_end(self):
        preds = torch.cat(self.validation_preds)
        labels = torch.cat(self.validation_labels)

        if hasattr(self, "current_pcc_matrices"):
            pcc_matrices = self.current_pcc_matrices
            sample_idx = 0
            epoch = self.current_epoch
            if sample_idx < len(pcc_matrices):
                self.visualize_pcc_matrix(
                    pcc_matrices[sample_idx].cpu().detach().numpy(), epoch
                )
        kappa = cohen_kappa_score(labels.cpu().numpy(), preds.cpu().numpy())
        self.log("val_kappa", kappa,prog_bar=True)
        self.validation_preds.clear()
        self.validation_labels.clear()

    def test_step(self, batch, batch_idx):
        loss, acc , preds , labels  = self.shared_step(batch, batch_idx, mode="test")
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        # self.save_test_accuracy(acc.item())
        return {"test_loss": loss, "test_acc": acc}

    def shared_step(self, batch, batch_idx, mode: str = "train", alpha=0.5):
        if mode == "train" and self.hparams.get("interaug", False):
            x,x_psd, y = interaug(batch)
        else:
            x,x_psd, y = batch
        y_hat = self.forward(x,x_psd)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.hparams.n_classes)
        preds = torch.argmax(y_hat, dim=-1)
        kappa = cohen_kappa_score(y.cpu().numpy(), preds.cpu().numpy())
        self.log(f"{mode}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{mode}_kappa", kappa, prog_bar=True, on_step=False, on_epoch=True)

        return loss, acc, preds, y

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return torch.argmax(self.forward(x), dim=-1)

    def on_test_epoch_end(self):

        print(f"Callback Metrics: {self.trainer.callback_metrics}")
        test_accs = self.trainer.callback_metrics.get("test_acc", None)
        test_kappa = self.trainer.callback_metrics.get("test_kappa", None)
        if test_accs is not None:
            final_test_acc = float(test_accs)
            final_test_kappa = float(test_kappa) if test_kappa is not None else None
            self.save_test_accuracy(final_test_acc, final_test_kappa)

        preds = torch.cat(self.test_preds).cpu().numpy()
        labels = torch.cat(self.test_labels).cpu().numpy()
        conf_matrix = confusion_matrix(labels, preds)
        csv_file = "confusion_matrix.csv"
        file_exists = os.path.exists(csv_file)
        with open(csv_file, mode="a", newline="") as file:  # 使用追加模式
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Test Round", "Class"] + list(range(conf_matrix.shape[1])))
            for i, row in enumerate(conf_matrix):
                writer.writerow([self.test_round, i] + list(row))
        self.test_preds.clear()
        self.test_labels.clear()



