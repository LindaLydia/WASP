# WASP

This is the repository for paper **Contrastive Private Data Synthesis via Weighted Multi-PLM Fusion** published in ICML2025 (see [link](https://icml.cc/virtual/2025/poster/44062)) and ICLR 2025 Workshop on Navigating and Addressing Data Problems for Foundation Models (see [link](https://openreview.net/forum?id=CPOFZJ8DlT)).


## 1. Environmental Setup
1. Please make sure that your cuda>=12.1.
2. Run the following command. Use `-i https://pypi.tuna.tsinghua.edu.cn/simple` to accelerate pip installation if necessary.
    ```bash
    conda create -n python3.9_torch2 python=3.9
    conda deactivate
    conda activate python3.9_torch2
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    piip install jsonlines tqdm transformers==4.41.2 torchtext==0.6.0 
    pip install argparse wandb matplotlib spacy pandas seaborn
    pip install accelerate==0.33.0 sentence-transformers==3.1.1
    pip install numpy==1.26.4
    conda install numpy==1.26.4
    pip install sentencepiece==0.1.96 datasets==2.19.1
    pip install bitsandbytes==0.44.1
    ```
3. If your cuda==11.8, use the following installation command.
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 # or use "conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia" to install from conda
    pip install transformers==4.41.2 tqdm jsonlines torchtext==0.6.0 
    pip install argparse wandb matplotlib spacy pandas seaborn
    pip install accelerate sentence-transformers==3.1.1
    pip install numpy==1.26.4
    conda install numpy==1.26.4
    pip install sentencepiece==0.1.96 datasets==2.19.1
    pip install bitsandbytes==0.44.1
    ```

### 2. Main Experiments
See `./src/run.sh`. The first instruction produces the results of WASP with IMDb dataset while the following 6 instruction produces the results for the most important PE series baselines (Aug-PE as we do on text tasks).

For real world private data, we randomly select samples from the training set of the related well-defined datasets (see `./src/data/`). For DP synthetic datasets, the starting data produced following [ZeroGen](https://github.com/jiacheng-ye/ZeroGen) without the help of real private sample information are placed within `./src/data_accumulate_start/`. Other data that are produced under the guidance of private samples will be stored automatically under `./src/data_accumulate/` after running the experiments.

### 3. Citation and Reference
```
@inproceedings{
    zou2025contrastive,
    title={{Contrastive Private Data Synthesis via Weighted Multi-PLM Fusion}},
    author={Tianyuan Zou and Yang Liu and Peng Li and Yufei Xiong and Jianqing Zhang and Jingjing Liu and Ye Ouyang and Xiaozhou Ye and Yaqin Zhang},
    booktitle={ICLR 2025 Workshop on Navigating and Addressing Data Problems for Foundation Models},
    year={2025},
    url={https://openreview.net/forum?id=CPOFZJ8DlT}
}
```
