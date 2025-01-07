# PrivateGenerateEnhancement

## Environmental Setup
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
