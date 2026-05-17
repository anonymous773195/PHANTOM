
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMDLoss(nn.Module):
    def __init__(self, kernel_bandwidths=(0.1, 1.0, 10.0)):
        super().__init__()
        self.kernel_bandwidths = kernel_bandwidths

    def _pairwise_sq_dist(self, x, y):
        
        xx = (x * x).sum(dim=1, keepdim=True)  # (B, 1)
        yy = (y * y).sum(dim=1, keepdim=True)  # (B, 1)
        dist = xx + yy.t() - 2.0 * x @ y.t()
        return dist.clamp(min=0.0)

    def _multi_scale_rbf(self, x, y):
        dist = self._pairwise_sq_dist(x, y)
        kernel = torch.zeros_like(dist)
        for bw in self.kernel_bandwidths:
            kernel = kernel + torch.exp(-dist / (2.0 * bw))
        return kernel

    def _mmd2(self, x, y):
        k_xx = self._multi_scale_rbf(x, x)
        k_yy = self._multi_scale_rbf(y, y)
        k_xy = self._multi_scale_rbf(x, y)
        B = x.size(0)
        if B < 2:
            return (k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        diag_mask = 1.0 - torch.eye(B, device=x.device)
        sum_xx = (k_xx * diag_mask).sum() / (B * (B - 1))
        sum_yy = (k_yy * diag_mask).sum() / (B * (B - 1))
        sum_xy = k_xy.sum() / (B * B)
        return sum_xx + sum_yy - 2.0 * sum_xy

    def forward(self, student_features, teacher_features):
        assert len(student_features) == len(teacher_features)
        mmd_total = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            s_pooled = s_feat.mean(dim=1)  # (B, D)
            t_pooled = t_feat.mean(dim=1)  # (B, D)
            mmd_total = mmd_total + self._mmd2(s_pooled, t_pooled)
        return mmd_total / len(student_features)


class KLDistillationLoss(nn.Module):

    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        T = self.temperature
        s_log_probs = F.log_softmax(student_logits / T, dim=1)
        t_probs = F.softmax(teacher_logits / T, dim=1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
        return kl


class PhasedKDLoss(nn.Module):

    def __init__(self, temperature=4.0, kernel_bandwidths=(0.1, 1.0, 10.0)):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss(kernel_bandwidths=kernel_bandwidths)
        self.kl_loss = KLDistillationLoss(temperature=temperature)

    def get_weights(self, current_epoch, total_epochs):
        progress = current_epoch / total_epochs
        if progress < 0.2:
            return 0.2, 0.4, 0.4
        elif progress < 0.7:
            return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
        else:
            t = (progress - 0.7) / 0.3
            alpha_distill = (1.0 / 3.0) * (1.0 - t) + 0.05 * t
            alpha_ce = 1.0 - 2.0 * alpha_distill
            return alpha_ce, alpha_distill, alpha_distill

    def forward(
        self,
        student_logits,
        teacher_logits,
        student_features,
        teacher_features,
        labels,
        current_epoch,
        total_epochs,
    ):
        alpha_ce, alpha_mmd, alpha_kl = self.get_weights(
            current_epoch, total_epochs,
        )

        l_ce = self.ce_loss(student_logits, labels)
        l_mmd = self.mmd_loss(student_features, teacher_features)
        l_kl = self.kl_loss(student_logits, teacher_logits)

        loss = alpha_ce * l_ce + alpha_mmd * l_mmd + alpha_kl * l_kl

        loss_dict = {
            "ce": l_ce.item(),
            "mmd": l_mmd.item(),
            "kl": l_kl.item(),
            "alpha_ce": alpha_ce,
            "alpha_mmd": alpha_mmd,
            "alpha_kl": alpha_kl,
            "total": loss.item(),
        }
        return loss, loss_dict
