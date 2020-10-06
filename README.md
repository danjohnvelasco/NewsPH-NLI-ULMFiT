# NewsPH-NLI-ULMFiT
This repository contains script to train and evaluate AWD LSTM on NewsPH-NLI dataset.

# Requirements
*  (fastai >= v2.0.15)[https://pypi.org/project/fastai/2.0.15/] 

# Reproducing Results
First, download the data and put it in the cloned repository:

```bash
# Create a new folder
mkdir NewsPH-NLI-ULMFiT/data

# NewsPH-NLI Dataset
wget https://s3.us-east-2.amazonaws.com/blaisecruz.com/datasets/hatenonhate/hatespeech_processed.zip
unzip newsph-nli.zip -d NewsPH-NLI-ULMFiT/data && rm newsph-nli.zip
```
### Textual Entailment Task

Here, textual entailment is treated as any classification task. To finetune, use the ```train.py``` script provided in this repository. Here's an example of finetuning a Filipino AWD LSTM model on the NewsPH-NLI dataset:

```bash
export DATA_DIR='Filipino-Text-Benchmarks/data/newsph-nli'n

python Filipino-Text-Benchmarks/train.py \
    --pretrained jcblaise/electra-tagalog-small-cased-discriminator \
    --train_data ${DATA_DIR}/train.csv \
    --valid_data ${DATA_DIR}/valid.csv \
    --test_data ${DATA_DIR}/test.csv \
    --data_pct 1.0 \
    --checkpoint model.pt \
    --do_train true \
    --do_eval true \
    --msl 128 \
    --optimizer adam \
    --batch_size 32 \
    --add_token [LINK],[MENTION],[HASHTAG] \
    --weight_decay 1e-8 \
    --learning_rate 2e-4 \
    --adam_epsilon 1e-6 \
    --warmup_pct 0.1 \
    --epochs 3 \
    --seed 42
```

This should give you the following results: 
```
Valid Loss 0.4980
Valid Acc 0.7655
Test Loss 0.5243
Test Accuracy 0.7467
```
