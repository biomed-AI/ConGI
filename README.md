



# Deciphering Spatial Domains by Integrating Histopathological Image and Transcriptomics via Contrastive Learning 




## Overview

The codes are coming soon


## Requirements
Please ensure that all the libraries below are successfully installed:

- **torch 1.7.1**
- CUDA Version 10.2.89
- scanpy 1.8.1
- louvain 0.7.0








## Run ConGI the example data.

```

python main.py --name her2+ E1

```


## output

The clustering cell labels will be stored in the dir [output](https://github.com/biomed-AI/ConGI) /dataname_pred.csv. 



## Datasets

 -  human HER2-positive breast tumor ST data https://github.com/almaan/her2st/.
 -  human cutaneous squamous cell carcinoma 10x Visium data (GSE144240).




## Citation

Please cite our paper:

```

@article{zengys,
  title={Deciphering Spatial Domains by Integrating Histopathological Image and Tran-scriptomics via Contrastive Learning },
  author={Yuansong Zeng1,#, Rui Yin3,#, Mai Luo1, Jianing Chen1, Zixiang Pan1, Yutong Lu1, Weijiang Yu1* , Yuedong Yang1,2*},
  journal={biorxiv},
  year={2022}
 publisher={Cold Spring Harbor Laboratory}
}

```
