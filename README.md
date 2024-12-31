# PrivateGenerateEnhancement

## Environmental Setup
1. Please make sure that your cuda>=12.1.
2. Run the following command.
    ```bash
    conda create -n python3.9_torch2 python=3.9
    conda deactivate
    conda activate python3.9_torch2
    conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    piip install jsonlines tqdm transformers==4.41.2 torchtext==0.6.0 
    pip install argparse wandb matplotlib spacy pandas seaborn
    pip install accelerate sentence-transformers==3.1.1
    pip install sentencepiece==0.1.96
    pip install -U bitsandbytes
    ```
3. If your cuda==11.8, use the following installation command.
    ```bash
    conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia # or use "pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118" to install from pip
    conda install numpy==1.26.4
    conda install transformers tqdm matplotlib jsonlines
    pip install torchtext==0.6.0 argparse wandb matplotlib spacy pandas seaborn
    pip install accelerate sentence-transformers==3.1.1
    ```
