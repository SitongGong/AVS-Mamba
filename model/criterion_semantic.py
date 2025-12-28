import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total ten frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total ten frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1)

    loss = nn.CrossEntropyLoss(reduction='none')(pred_mask, ten_gt_masks)  # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1)  # [bs*10]
    loss = loss * gt_temporal_mask_flag  # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


def contrastive_loss_modi(origin_audio_feat, audio_feat, tau):
    # origin_audio_feat: bs, t, dim
    # audio_feat: bs, t, dim
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau))
    logit_scale = torch.clamp(logit_scale.exp(), max=100)          # 可学习参数调节

    logits = torch.arange(1).long().to(audio_feat.device)

    bs, t = audio_feat.shape[0], audio_feat.shape[1]
    origin_audio_feat = origin_audio_feat.flatten(0, 1)            # bs * t, dim
    audio_feat = audio_feat.flatten(0, 1)              # bs * t, dim

    origin_audio_feat = F.normalize(origin_audio_feat, dim=-1)
    audio_feat = F.normalize(audio_feat, dim=-1)

    loss_origin_audio = origin_audio_feat @ audio_feat.t()         # bs * t, bs * t
    loss_audio_feat = audio_feat @ origin_audio_feat.t()           # bs * t, bs * t
    # loss_origin_audio = origin_audio_feat.reshape(bs, t, -1)          # bs, time, bs * time
    
    cross_entropy_loss = nn.CrossEntropyLoss()

    contrastive_loss1 = 0

    for idx in range(loss_audio_feat.shape[0]):
        if idx // t == 0:
            cat_audio_feat = torch.cat((loss_audio_feat[idx][idx].unsqueeze(0), loss_audio_feat[idx][(idx // t + 1) * t:]))
        elif idx // t == bs - 1:
            cat_audio_feat = torch.cat((loss_audio_feat[idx][idx].unsqueeze(0), loss_audio_feat[idx][: (idx // t) * t]))
        else:
            cat_audio_feat = torch.cat((loss_audio_feat[idx][idx].unsqueeze(0), loss_audio_feat[idx][: (idx // t) * t], loss_audio_feat[idx][(idx // t + 1) * t:]))

        contrastive_loss1 = contrastive_loss1 + cross_entropy_loss(cat_audio_feat.unsqueeze(0) * logit_scale, logits)

    contrastive_loss2 = 0

    for idx in range(loss_origin_audio.shape[0]):
        if idx // t == 0:
            cat_audio_feat = torch.cat((loss_origin_audio[idx][idx].unsqueeze(0), loss_origin_audio[idx][(idx // t + 1) * t:]))
        elif idx // t == bs - 1:
            cat_audio_feat = torch.cat((loss_origin_audio[idx][idx].unsqueeze(0), loss_origin_audio[idx][: (idx // t) * t]))
        else:
            cat_audio_feat = torch.cat((loss_origin_audio[idx][idx].unsqueeze(0), loss_origin_audio[idx][: (idx // t) * t], loss_origin_audio[idx][(idx // t + 1) * t:]))
        
        contrastive_loss2 = contrastive_loss2 + cross_entropy_loss(cat_audio_feat.unsqueeze(0) * logit_scale, logits)

    contrastive_loss = contrastive_loss1 / (bs * t) + contrastive_loss2 / (bs * t)

    return contrastive_loss


def Loss(pred_mask, gt_mask, weight_dict, audio_feat, mask_feature, gt_temporal_mask_flag):
    # pred_mask: l, bs * t, dim, h, w          # 改为直接用dice_loss和focal_loss的
    # mask_features: bs * t, dim, h * w
    # gt_mask: bs x t, 1, h, w
    
    rec_loss = contrastive_loss_modi(audio_feat, mask_feature, tau=0.07)

    interm_masks = pred_mask[: -1]
    pred_mask = pred_mask[-1]   # bs * t, nclass, h, w

    one_mask = torch.ones_like(gt_mask)
    norm_gt_mask = torch.where(gt_mask > 0, one_mask, gt_mask)

    total_loss = 0
    bce_loss = F10_IoU_BCELoss(pred_mask, gt_mask, gt_temporal_mask_flag)

    interm_loss = 0
    for interm_mask in interm_masks:
        interm_loss += F10_IoU_BCELoss(interm_mask, gt_mask, gt_temporal_mask_flag) * weight_dict['bce_loss']

    total_loss = interm_loss * weight_dict['interm_loss'] + weight_dict['bce_loss'] * bce_loss + weight_dict['rec_loss'] * rec_loss

    loss_dict = {'bce_loss': bce_loss, 'interm_loss': interm_loss, 'rec_loss': rec_loss, 'total_loss': total_loss}

    return total_loss, loss_dict