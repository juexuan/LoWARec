# LoWARec
This is the pytorch implementation of our paper

[pdf]: https://link.springer.com/chapter/10.1007/978-981-92-0363-5_15	"pdf"

```
Low-Rank Guided Attention with Wavelet Augmentation for Sequential Recommendation.
```


## Quick Start


### How to train LoWARec
- Note that pretrained model (.pt) and train log file (.log) will saved in `src/output`
- `train_name`: name for log file and checkpoint file

```bash
python main.py  --model_typ lowarec
                --data_name [DATASET] \
                --proj_dropout [dropout rate] \
                --lr [LEARNING_RATE] \
                --alpha [ALPHA] \ 
                --c [C] \
                --proj_n_heads [N_HEADS] \
                --dim_proj [P] \
                --train_name [LOG_NAME]
```
- Example for LastFM
```bash
python main.py  --model_typ lowarec 
                --data_name LastFM 
                --proj_dropout 0.5 
                --lr 0.001 
                --alpha 0.7 
                --c 3 
                --proj_n_heads 4  
                --dim_proj 8 
                --train_name lowarec_LastFM_res
```

## Acknowledgement

Any scientific publications that use our codes and datasets should cite the following paper as the reference:

```tex
@inproceedings{LoWARec,
  author       = {Hao Li and
                  Mingxing Shao and
                  Tiancheng Zhang and
                  Minghe Yu and
                  Xue Geng and
                  Ge Yu},
  title        = {Low-Rank Guided Attention with Wavelet Augmentation for Sequential
                  Recommendation},
  booktitle    = {International Conference on Database Systems for Advanced Applications},
  pages        = {240--257},
  publisher    = {Springer},
  year         = {2026},
  doi          = {10.1007/978-981-92-0363-5\_15},
}
```

