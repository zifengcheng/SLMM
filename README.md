# Learning to Classify Open Intent via Soft Labeling and Manifold Mixup

This repo contains the code of our TASLP'2022 paper:

Learning to Classify Open Intent via Soft Labeling and Manifold Mixup

## Requirements

- Python 3.6
- PyTorch 1.8.0
- transformers 2.8.0
- pytorch_pretrained_bert 0.6.2

## Model Preparation
Get the pre-trained [BERT](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) model and convert it into [Pytorch](https://huggingface.co/transformers/converting_tensorflow_models.html). 

Set the path of the uncased-bert model (parameter "bert_model" in init_parameter.py).

## Quick Start  

Run our model:

bash run_0.25_oos.sh

If you are insterested in this work, and want to use the codes or results in this repository, please **star** this repository and **cite** by:
```
@article{Cheng22,
  author={Cheng, Zifeng and Jiang, Zhiwei and Yin, Yafeng and Wang, Cong and Gu, Qing},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={Learning to Classify Open Intent via Soft Labeling and Manifold Mixup}, 
  year={2022},
  volume={30},
  pages={635-645},
  doi={10.1109/TASLP.2022.3145308}
}
```
## Acknowledgments
We thank all authors from this paper: 'Deep Open Intent Classification with Adaptive Decision Boundary'. We adopt many codes from their projects. Thank a lot!

