# LoWARec
This is the official source code for LoWARec.


## Quick Start


### How to train LoWARec
- Note that pretrained model (.pt) and train log file (.log) will saved in `src/output`
- `train_name`: name for log file and checkpoint file

```
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
```
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
