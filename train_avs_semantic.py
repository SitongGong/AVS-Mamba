import torch
import time
import torch.nn
import os
import random
import numpy as np
# from mmcv import Config
import argparse

import logging

from model.compformer_semantic import CompFormer
from model.criterion_semantic import Loss
from dataloader import build_dataset
from utils.compute_color_metrics import calc_color_miou_fscore
# from utility import mask_iou, Eval_Fmeasure
from utils.loss_util import LossUtil
from utils import pyutils

from torch.utils.tensorboard import SummaryWriter

# 引入vggish对音频进行预处理
from model.vggish import vggish_input
def _preprocess(x):
        if isinstance(x, str):
            x = vggish_input.waveform_to_examples(x)
            return x
        else:
            batch_num = len(x)
            audio_fea_list = []
            for xx in x:
                if isinstance(xx, str):
                    xx = vggish_input.wavfile_to_examples(
                        xx)  # [5 or 10, 1, 96, 64]
                    #! notice:
                    if xx.shape[0] != 10:
                        new_xx = torch.zeros(10, 1, 96, 64)
                        new_xx[:xx.shape[0]] = xx
                        audio_fea_list.append(new_xx)
                    else:
                        audio_fea_list.append(xx)

            # [bs, 10, 1, 96, 64]
            audio_fea = torch.stack(audio_fea_list, dim=0)
            audio_fea = audio_fea.view(
                batch_num * 10, xx.shape[1], xx.shape[2], xx.shape[3])  # [bs*10, 1, 96, 64]
            audio_fea = audio_fea.cuda()
            return audio_fea

def getLogger(log_file, name, fmt='%(asctime)s %(levelname)s ==> %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def train(args):
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    # dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, args.dir_name)):
        os.mkdir(os.path.join(args.log_dir, args.dir_name))
    log_file = os.path.join(args.log_dir, args.dir_name, f'{log_name}.log')
    logger = getLogger(log_file, __name__)
    # logger.info(f'Load config from {args.cfg}')
    writer = SummaryWriter(args.log_dir)

    # model 
    vggish=dict(       # 音频编码器模型
        freeze_audio_extractor=True,
        pretrained_vggish_model_path=args.audio_pretrained,
        preprocess_audio_to_log_mel=False,
        postprocess_log_mel_with_pca=False,
        pretrained_pca_params_path=None)
    # mask2former模型
    model = CompFormer(d_model=args.d_model, 
                        nheads=args.nheads, 
                        num_query=args.num_query, 
                        vggish_config=vggish, 
                        audio_dim=128,
                        num_frame=args.mask_num,
                        mask_dim=args.mask_dim,
                        dim_feedforward=args.dim_feedforward,
                        num_classes=args.num_class,
                        scale_factor=args.scale_factor,
                        vision_checkpoint=args.vision_pretrained,
                        controller_layers=3,
                        dynamic_mask_channels=8)
    if args.resume:     # 加载之前的训练模型
        model_dict = model.state_dict()
        pretrained_state_dicts = torch.load(args.model_weight)        # 预训练模型的权重
        state_dict = {k: v for k, v in pretrained_state_dicts.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model).cuda()       # 分布式训练Missing key(s) in state_dict: "visual_fusion_block.multihead_attn.in_proj_weight", "visual_fusion_block.multihead_attn.in_proj_bias", "visual_fusion_block.multihead_attn.out_proj.weight", "visual_fusion_block.multihead_attn.out_proj.bias", "mta_layer.cross_attn_layer.cross_attn.in_proj_weight", "mta_layer.cross_attn_layer.cross_attn.in_proj_bias", "mta_layer.cross_attn_layer.cross_attn.out_proj.weight", "mta_layer.cross_attn_layer.cross_attn.out_proj.bias", "mta_layer.cross_attn_layer.norm1.weight", "mta_layer.cross_attn_layer.norm1.bias", "mta_layer.ffn.linear1.weight", "mta_layer.ffn.linear1.bias", "mta_layer.ffn.linear2.weight", "mta_layer.ffn.linear2.bias", "mta_layer.ffn.norm2.weight", "mta_layer.ffn.norm2.bias", "mta_layer.cross_attn_layers.0.cross_attn.in_proj_weight", "mta_layer.cross_attn_layers.0.cross_attn.in_proj_bias", "mta_layer.cross_attn_layers.0.cross_attn.out_proj.weight", "mta_layer.cross_attn_layers.0.cross_attn.out_proj.bias", "mta_layer.cross_attn_layers.0.norm1.weight", "mta_layer.cross_attn_layers.0.norm1.bias", "mta_layer.ffn_layers.0.linear1.weight", "mta_layer.ffn_layers.0.linear1.bias", "mta_layer.ffn_layers.0.linear2.weight", "mta_layer.ffn_layers.0.linear2.bias", "mta_layer.ffn_layers.0.norm2.weight", "mta_layer.ffn_layers.0.norm2.bias".
    model.train()
    logger.info("Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
    
    # dataset
    train_dataset = build_dataset(args.dataset, 'train', args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.train_batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_works,
                                                   pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.train_epochs

    N_CLASSES = train_dataset.num_classes

    val_dataset = build_dataset(args.dataset, 'test', args)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.val_batch_size,
                                                 shuffle=False,
                                                 num_workers=args.num_works,
                                                 pin_memory=True)
    
    # optimizer
    weight_dict = {'bce_loss': 1, 'interm_loss': 1, 'rec_loss': 0}
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    loss_util = LossUtil(weight_dict)

    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    max_miou = 0
    miou_noBg_list = []
    Fscore_list = []
    Fscore_noBg_list = []
    for epoch in range(args.train_epochs):
        # if epoch == args.freeze_epochs:
        #   model.module.freeze_backbone(False)
        if epoch == 0:
            model.module.vision_backbone_frozen(True)
        total_loss = 0
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            gt_temporal_mask_flag = gt_temporal_mask_flag.cuda()

            imgs = imgs.cuda()
            # audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask_num = 10
            mask = mask.view(B * mask_num, 1, H, W)
            # audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])
            
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(B * frame)  # [B*T]
            gt_temporal_mask_flag = gt_temporal_mask_flag.view(B * frame)  # [B*T]
            
            audio = _preprocess(audio)      # bs * 10, 1, 96, 64

            pred_mask, mask_features, mask_feature, audio_feat = model(audio, imgs, mask.reshape(B, mask_num, H, W), vid_temporal_mask_flag)       # pred_mask: l, bs * t, 1, h, w     mask_features: bs * t, c, h, w
            loss, loss_dict = Loss(pred_mask, mask, weight_dict, audio_feat, mask_feature, gt_temporal_mask_flag)
            loss_util.add_loss(loss, loss_dict)          # 保存并记录每个batch的loss值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            global_step += 1
            if (global_step - 1) % 20 == 0:
                train_log = 'Iter:%5d/%5d, %slr: %.6f' % (
                    global_step - 1, max_step, loss_util.pretty_out(), optimizer.param_groups[0]['lr'])
                logger.info(train_log)

        writer.add_scalar('total_loss', total_loss / n_iter, epoch)

        # Validation:
        model.eval()
        miou_pc = torch.zeros((N_CLASSES))
        Fs_pc = torch.zeros((N_CLASSES))  # f-score per class (total sum)
        cls_pc = torch.zeros((N_CLASSES))  # count per class
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]
                imgs, audio, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data

                vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
                gt_temporal_mask_flag = gt_temporal_mask_flag.cuda()

                imgs = imgs.cuda()
                # audio = audio.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B * frame, C, H, W)
                mask = mask.view(B * frame, H, W)
                # audio = audio.view(-1, audio.shape[2], audio.shape[3], audio.shape[4])

                #! notice
                vid_temporal_mask_flag = vid_temporal_mask_flag.view(B * frame)  # [B*T]
                gt_temporal_mask_flag = gt_temporal_mask_flag.view(B * frame)  # [B*T]

                audio = _preprocess(audio)      # bs * 10, 1, 96, 64

                # [bs*5, 1, 224, 224]
                output, _, _, _ = model(audio, imgs, mask.reshape(B, frame, H, W), vid_temporal_mask_flag)
                output = output[-1]          # bs * t, 1, h, w
                # miou = mask_iou(output.squeeze(1), mask)             # 计算miou值     bs * t, h, w
                # Fscore = Eval_Fmeasure(output.squeeze(1), mask)
                
                _miou_pc, _fscore_pc, _cls_pc, _ = calc_color_miou_fscore(output, mask)
                # compute miou, J-measure
                miou_pc += _miou_pc
                cls_pc += _cls_pc
                # compute f-score, F-measure
                Fs_pc += _fscore_pc
                
                # avg_meter_miou.add({'miou': miou})
                # avg_meter_fscore.add({'F_score': Fscore})

            miou_pc = miou_pc / cls_pc
            logger.info(f"[miou] {torch.sum(torch.isnan(miou_pc)).item()} classes are not predicted in this batch")
            miou_pc[torch.isnan(miou_pc)] = 0
            miou = torch.mean(miou_pc).item()
            miou_noBg = torch.mean(miou_pc[:-1]).item()
            f_score_pc = Fs_pc / cls_pc
            logger.info(f"[fscore] {torch.sum(torch.isnan(f_score_pc)).item()} classes are not predicted in this batch")
            f_score_pc[torch.isnan(f_score_pc)] = 0
            f_score = torch.mean(f_score_pc).item()
            f_score_noBg = torch.mean(f_score_pc[:-1]).item()

            if miou > max_miou or f_score + miou > max_fs + max_miou:
                model_save_path = os.path.join(args.checkpoint_dir, '%s_best.pth' % (args.session_name))
                torch.save(model.module.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s' % model_save_path)

            miou_list.append(miou)
            miou_noBg_list.append(miou_noBg)
            max_miou = max(miou_list)
            max_miou_noBg = max(miou_noBg_list)
            Fscore_list.append(f_score)
            Fscore_noBg_list.append(f_score_noBg)
            max_fs = max(Fscore_list)
            max_fs_noBg = max(Fscore_noBg_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}, Miou(no bg): {}, maxMiou (no bg): {} '.format(
                epoch, miou, max_miou, miou_noBg, max_miou_noBg)
            val_log += ' Fscore: {}, maxFs: {}, Fscore(no bg): {}, max Fscore (no bg): {}'.format(
                f_score, max_fs, f_score_noBg, max_fs_noBg)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('cfg', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str, default='', help='log dir')
    parser.add_argument('--dir_name', type=str, default='V2', help='log dir name')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='dir to save checkpoints')
    parser.add_argument("--session_name", default="V2", type=str, help="the V2 setting")
    parser.add_argument('--vision_pretrained', help='dir to vision checkpoints', 
                        default='pretrained_backbones/pvt_v2_b5.pth', type=str)

    parser.add_argument('--lr', help='learning rate', default=2e-5, type=float)
    parser.add_argument('--audio_pretrained', help='dir to audio checkpoints', 
                        default='pretrained_backbones/vggish-10086976.pth', type=str)
    parser.add_argument('--dataset', default='V2Dataset', type=str, help='the dataset to train')        # 目前先支持MS3和S4
    parser.add_argument('--dir_base', default='AVSBench-semantic', type=str)
    parser.add_argument('--mask_num', default=10, type=int)
    parser.add_argument('--crop_img_and_mask', default=True, type=bool)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--meta_csv_path', default='AVSBench-semantic/metadata.csv', type=str)
    parser.add_argument('--label_idx_path', default='AVSBench-semantic/label2idx.json', type=str)
    parser.add_argument('--num_class', default=71, type=int)

    parser.add_argument('--train_epochs', default=30, type=int)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--val_batch_size', default=16, type=int)
    parser.add_argument('--num_works', default=8, type=int)
    parser.add_argument('--loss_type', default='dice', type=str)

    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_query', default=1, type=int)
    parser.add_argument('--mask_dim', default=256, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--scale_factor', default=4, type=int)

    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--model_weight', default=None, type=str)

    args = parser.parse_args()
    train(args)
