# Segment Anything (SAM) from scatch 
From scratch implementation of the paper: "[Segment Anything](https://arxiv.org/abs/2304.02643)" ICCV 2023

### Clone and install dependencies
``` 
git clone https://github.com/aldipiroli/sam_from_scratch
pip install -r requirements.txt
``` 
### Train 
``` 
cd python 
python train.py
```

> Note: this repository is just for learning purposes and does not aim at the performance of the [original repository](https://github.com/facebookresearch/segment-anything). Additionally, some features from the original model are not implemented (i.e., box/test prompting), and the training dataset is the [Pascal VOC 2012 dataset](https://cocodataset.org/index.htm#home).
