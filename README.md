# AVS-Mamba
This is the official code for the AVS-Mamba, which is accepted by IEEE Transactions on Multimedia

## Datasets
Please download AVSBench-object (including MS3 and S4) and AVSBench-semantic following this repo [AVSBench](https://github.com/OpenNLPLab/AVSBench).

## Training and Inference
For training, please refer to the following commands:
```bash
cd AVS-Mamba

python train_ablation.py \
            --use_vision_backbone 'PVTv2' \
            --session_name 'MS3' \
            --data_dir 'Multi-sources' \
            --train_epochs 120 \
            --train_batch_size 2 \
            --unfreeze_epoch 20 \
            --if_use_encoder \
            --if_use_decoder \
            --use_inter_decoder \
            --use_intra_decoder \
            --use_temporal_encoder \
            --use_spatial_encoder \
            --if_use_cmfpn \
            --use_temporal_mamba \
            --use_avfusion \
            --log_dir 'avsmamba/ms3' \
            --checkpoint_dir 'avsmamba/ms3' \
            --scan_order 8
```

For inference, you can use following commands:
```bash
cd AVS-Mamba
python train_ablation.py \
        --session_name 'MS3' \
        --use_vision_backbone 'PVTv2' \
        --resume \
        --if_use_encoder \
        --if_use_decoder \
        --use_inter_decoder \
        --use_intra_decoder \
        --use_temporal_encoder \
        --use_spatial_encoder \
        --if_use_cmfpn \
        --use_temporal_mamba \
        --use_avfusion \
        --scan_order 8 \
        --eval_only True \
        --val_batch_size 1 \
        --data_dir 'your own checkpoint path' \
```

## Citation
If you find this project useful in your research, please consider citing:

```
@article{gong2025avs,
  title={Avs-mamba: Exploring temporal and multi-modal mamba for audio-visual segmentation},
  author={Gong, Sitong and Zhuge, Yunzhi and Zhang, Lu and Wang, Yifan and Zhang, Pingping and Wang, Lijun and Lu, Huchuan},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgement
We sincerely thank the following works for their valuable contributions: [TPAVI](https://github.com/OpenNLPLab/AVSBench), [AVSegFormer](https://github.com/vvvb-github/AVSegFormer), [Vim](https://github.com/hustvl/Vim), [VMamba](https://github.com/MzeroMiko/VMamba)
