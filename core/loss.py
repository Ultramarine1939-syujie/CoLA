# Code for CVPR'21 paper:
# [Title]  - "CoLA: Weakly-Supervised Temporal Action Localization with Snippet Contrastive Learning"
# [Author] - Can Zhang*, Meng Cao, Dongming Yang, Jie Chen and Yuexian Zou
# [Github] - https://github.com/zhang-can/CoLA

import torch
import torch.nn as nn

# 定义动作损失函数
class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        # 将标签归一化
        label = label / torch.sum(label, dim=1, keepdim=True)
        # 计算二分类交叉熵损失
        loss = self.bce_criterion(video_scores, label)
        return loss

# 定义SniCo损失函数
class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        # 对输入进行归一化
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        # 计算正样本和负样本的内积
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # 计算交叉熵损失
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):
        # 计算两个对比组的SniCo损失
        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )

        # 总损失为两个对比组的SniCo损失之和
        loss = HA_refinement + HB_refinement
        return loss
        

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def forward(self, video_scores, label, contrast_pairs):
        # 计算动作损失
        loss_cls = self.action_criterion(video_scores, label)
        # 计算SniCo损失
        loss_snico = self.snico_criterion(contrast_pairs)
        # 计算总损失
        loss_total = loss_cls + 0.01 * loss_snico

        loss_dict = {
            'Loss/Action': loss_cls,
            'Loss/SniCo': loss_snico,
            'Loss/Total': loss_total
        }

        return loss_total, loss_dict
