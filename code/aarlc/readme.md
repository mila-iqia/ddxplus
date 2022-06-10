1. To test the method on DDXPlus disease sets,
   1. cd ./ddxplus_code
   2. CUDA_VISIBLE_DEVICES=0 python3 main.py --seed 42 --train_data_path "release_train_patients.zip"  --val_data_path "release_validate_patients.zip" --train --trail 1 --nu 2.826 --mu 1.0 --lr 0.000352 --lamb 0.99 --gamma 0.99 --eval_batch_size 4139  --batch_size 2657  --EPOCHS 100 --MAXSTEP 30 --patience 20 --eval_on_train_epoch_end
   3. CUDA_VISIBLE_DEVICES=0 python3 main.py --seed 42 --train_data_path "release_train_patients.zip"  --val_data_path "release_validate_patients.zip" --train --trail 1 --nu 3.337 --mu 1.0 --lr 0.0005175 --lamb 0.97 --gamma 0.99  --eval_batch_size 4139  --batch_size 2657  --EPOCHS 100 --MAXSTEP 30 --patience 20 --eval_on_train_epoch_end  --no_differential
 

The citation for our paper is:
```
@misc{https://doi.org/10.48550/arxiv.2112.00733,
  doi = {10.48550/ARXIV.2112.00733},
  url = {https://arxiv.org/abs/2112.00733},
  author = {Yuan, Hongyi and Yu, Sheng},
  title = {Efficient Symptom Inquiring and Diagnosis via Adaptive Alignment of Reinforcement Learning and Classification},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
