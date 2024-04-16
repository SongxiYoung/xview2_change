# Self-supervised learning for change detection

This repo aims at using self-supervised learning for extracting informative image features for bi-temporal change detection without human labels.

## Dataset

- Building image patches: `/home/bpeng/mnt/mnt242/scdm_data/xBD/xbd_disasters_building_polygons_neighbors`
    - csv files with list of building patch (IDs, damage labels): `./csvs_buffer/train_tier3_test_hold_wo_unclassified.csv`
    - actual building image patches: `./images_buffer` 
    
**NOTE**: **DO NOT OPEN the directory directly on your PC since it contains a huge number of images**

## Code

- Main script:
    - `train_simclr.py`: A reproduced code for the model SimCLR
    - `train_stcrl.py`: The self-developed model derived from SimCLR incorporating geographic and temporal knowledge into self-supervised learning as discussed in the IGARSS 2021 paper.
    - `train_stcrl_FixMatch.py`: The stcrl model with FixMatch method.
- Models
    - `resnet.py`: backbone of both `SimCLR` and `STCRL`.
    - `simclr_net`: the network of `SimCLR`
    - `stcrl_net`: the network of `STCRL`
- Utils
    - `utils.py`: utility tools for data processing, result logging, etc.
    - `dataproc_double.py`: data processing of the **xview2** dataset for PyTorch deep learning.


## Reference

- Peng, Bo, Qunying Huang, and Jinmeng Rao. "Spatiotemporal Contrastive Representation Learning for Building Damage Classification." 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS. IEEE, 2021.
- Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.
- Zbontar, Jure, et al. "Barlow twins: Self-supervised learning via redundancy reduction." International Conference on Machine Learning. PMLR, 2021.
- Sohn, Kihyuk, et al. "Fixmatch: Simplifying semi-supervised learning with consistency and confidence." Advances in neural information processing systems 33 (2020): 596-608.
