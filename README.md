# NewsPH-NLI-ULMFiT
This repository contains script to train and evaluate AWD LSTM on NewsPH-NLI dataset.

# Requirements
*  [fastai >= v2.0.15](https://pypi.org/project/fastai/2.0.15/)
*  NVIDIA GPU (all experiments were done in Colab w/ Tesla T4)

# Reproducing Results
First, clone the repository and download the data:

```bash
# Clone this repository
git clone https://github.com/danjohnvelasco/NewsPH-NLI-ULMFiT

cd NewsPH-NLI-ULMFiT

# Install gdown
pip install gdown

# Create a new folder
mkdir data

# Download NewsPH-NLI Dataset (preprocessed)
gdown --id 1-qOfNQy-piiaz8BcDfnS-ILlYio5S_g2

# Unzip
unzip newsph-nli-preprocessed.zip -d data
```

## Download language model fine-tuned on NewsPH-NLI
```bash
# Make directory
mkdir models

# Download data
gdown --id 1-PI65kBGD0i2hE3KL5hjjCGDt_mMFKMs

# Unzip
unzip finetuned.zip -d models

# Finally
You should see two files: lm_fintuned_enc.pth (encoder) and news_vocab.pkl (vocab). 
This will be used later in classifier finetuning.
```

### Textual Entailment Task

Here, textual entailment is treated as any classification task. To fine-tune, use the ```train.py``` script provided in this repository. Here's an example of fine-tuning a Filipino AWD-LSTM model on the NewsPH-NLI dataset:

```bash
python train.py \
    --pretrained_path "lm_fintuned_enc" \
    --vocab_path "models/news_vocab.pkl" \
    --checkpoint "model" \
    --train_data data/train.csv \
    --valid_data data/valid.csv \
    --test_data data/test.csv \
    --do_train \
    --do_eval \
    --batch_size 128 \
    --weight_decay 0.1 \
    --seed 42 \
    --lr_max 10e-3 \
    --epochs "4;2;2;2;1" \
    --data_pct 1.0
```

This should give you the following results: 
```
Valid Loss 0.2685 | Valid Accuracy 0.8911
Test Loss 0.2589 | Test Accuracy 0.8937
```
