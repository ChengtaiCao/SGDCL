# SGDCL: Semantic-Guided Dynamic Correlation Learning for Explainable Autonomous Driving

![](https://img.shields.io/badge/python-3.9-green)
![](https://img.shields.io/badge/torch-2.0.1-green)
![](https://img.shields.io/badge/cudatoolkit-11.7-green)

This repo provides a reference implementation of “SGDCL: Semantic-Guided Dynamic Correlation Learning for Explainable Autonomous Driving”. 

# Abstract
  By learning expressive representations, deep learning (DL) has revolutionized autonomous driving (AD). Despite significant advancements, the inherent opacity of DL models engenders public distrust, impeding their widespread adoption. For explainable autonomous driving, current studies primarily concentrate on extracting features from input scenes to predict driving actions and their corresponding explanations. However, these methods underutilize semantics and correlation information within actions and explanations (collectively called categories in this work), leading to suboptimal performance. To address this issue, we propose Semantic-Guided Dynamic Correlation Learning (SGDCL), a novel approach that effectively exploits semantic richness and dynamic interactions intrinsic to categories. SGDCL employs a semantic-guided learning module to obtain category-specific representations and a dynamic correlation learning module to adaptively capture intricate correlations among categories. Additionally, we introduce an innovative loss term to leverage fine-grained co-occurrence statistics of categories for refined regularization. We extensively evaluate SGDCL on two well-established benchmarks, demonstrating its superiority over seven state-of-the-art baselines and a large vision-language model. SGDCL significantly promotes explainable autonomous driving with up to $15.3\%$ performance improvement and interpretable attention scores, bolstering public trust in AD.


# Dataset
## BDD-OIA
BDD-OIA, a subset of [BDD100K](https://www.vis.xyz/bdd100k), contains 22,924 video frames, each annotated with 4 action decisions and 21 human-defined explanations.  
Head to [X-OIA](https://twizwei.github.io/bddoia_project) to download the dataset.  
Following [Xu et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Explainable_Object-Induced_Action_Decision_for_Autonomous_Vehicles_CVPR_2020_paper.pdf), only the final frame of each video clip is used, leading to a training set of 16,082 images, a validation set of 2,270 images and a test set of 4,572 images.
## PSI
PSI includes 11,902 keyframes, each annotated with 3 actions and explanations provided in natural language.  
Head to [PSI](http://pedestriandataset.situated-intent.net) to download the dataset.  
Following [Zhang et al.](https://ieeexplore.ieee.org/document/9991055), all samples are split into training, validation, and test sets with a ratio of 7/1/2.

# Environmental Settings
Our experiments are conducted on Ubuntu 22.04, a single NVIDIA GeForce RTX 3080 GPU, 64GB RAM, and Intel i70-11700K. SDDCL is implemented by `Python 3.9`, `PyTorch 2.0.1`, and `Cuda 11.7`.

**Step 1**: Install [Anaconda](https://www.anaconda.com)

**Step 2**: Create a virtual environment and install the required packages
```shell
# create a new environment
conda create -n SGDCL python=3.9

# activate environment
conda activate SGDCL

# install Pytorch
pip install torch torchvision torchaudio

# install other required packages
pip install -r requirements.txt
```

# Usage
**Step 0**: Create some folder
```shell
mkdir bddoia log save_model weight
```

**Step 1**: Download datasets: put BDD-OIA data in bddoia.

**Step 2**: Download pre-trained weight from [NLE-DM](https://github.com/lab-sun/NLE-DM/tree/main) and put it in folder "weight".

**Step 3**: Generate adjacency information.
```shell
python GenAdj.py
```

**Step 4**: Generate sentence embeddings.
```shell
python LabelSemantic.py
```

**Step 5**: Train model.
```shell
python train_OIA.py
```

**Step 6**: Test model.
```shell
python prediction_OIA.py
```

# Default hyperparameter settings

Unless otherwise specified, we use the following default hyperparameter settings.

Param|Value|Param|Value
:---|---:|:---|---:
learning rate|0.001|batch_size|2
momentum|0.9|epoches|50
weight decay|0.0001|cross attention dim|8
GNN hidden dim|8|GNN output dim|16
GNN attention head|8|classifier hidden dim|64

# Acknowledgement
We would like to thank the great work: [NLE-DM](https://ieeexplore.ieee.org/document/10144484) and its code [repo](https://github.com/lab-sun/NLE-DM/tree/main).

# Cite
If you find our paper or code are useful for your research, please consider citing us:

```bibtex
@inproceedings{cao2024sdgcl, 
  author = {Chengtai Cao and Xinhong Chen and Jianping Wang and Qun Song and Rui Tan and Yung-Hui Li}, 
  title = {SGDCL: Semantic-Guided Dynamic Correlation Learning for Explainable Autonomous Driving}, 
  booktitle={International Joint Conference on Artificial Intelligence},
  pages={596--604},
  year = {2024}
}
```

# Contact
Any comments and feedbacks are appreciated.