import pickle
from fastai.text.all import *
import sys
import os


# This sets the random seed manually
# Run this before creating a learner
def set_random_seed(seed):
    # python RNG
    import random
    random.seed(seed)

    # pytorch RNGs
    import torch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # numpy RNG
    import numpy as np
    np.random.seed(seed)


def evaluate(args, data_path):
    print('\n' + '=' * 50, '\nEVALUATING MODE', '\n' + '=' * 50)

    print(f"loading {data_path}...")
    test_df = pd.read_csv(data_path)

    # Add dummy train set
    print("setting up data...")
    dummy_df = test_df.iloc[:5].copy() # arbitrary number
    dummy_df['is_valid'] = False # dummy training data
    test_df = pd.concat([dummy_df, test_df])

    # load vocab
    print("loading vocab...")
    with open(args.vocab_path, 'rb') as f:
        news_vocab = pickle.load(f)

    # Create test dataloader
    print("creating test dataloader...")
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    test_dl = DataBlock(blocks=(TextBlock.from_df('text', min_freq=2, vocab=news_vocab), CategoryBlock),
                        get_x=ColReader('text'),
                        get_y=ColReader('label'),
                        splitter=ColSplitter()).dataloaders(test_df, bs=128, num_workers=0)
    print("test dataloader DONE!...")
    sys.stdout = old_stdout  # reset old stdout

    # Create model
    print("creating model...")
    learn = text_classifier_learner(
        test_dl, AWD_LSTM, wd=0.1, metrics=accuracy).to_fp16()

    # If want to load classifier model
    if args.classifier_path:
        print("loading classifier model...")
        learn.load(args.classifier_path)
    else: # Model from training run
        print(f"loading checkpoint model ({args.checkpoint})")
        learn.load(args.checkpoint)

    print("running validate()...")
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    res = learn.validate()
    sys.stdout = old_stdout  # reset old stdout

    test_loss = res[0]
    test_acc = res[1]

    return test_loss, test_acc
    
# training script
def run_finetuning(args):

    # Set seed
    print(f"Setting random seed: {args.seed}")
    set_random_seed(args.seed)

    # Training
    if args.do_train:
        print('\n' + '=' * 50, '\nCONFIGURE FINETUNING SETUP', '\n' + '=' * 50)
        
            # Load data
        if args.preloaded_data_path:
            # Contains training and validation set
            print(f"loading preloaded data from {args.preloaded_data_path}")
            classifier_dataloader = torch.load(args.preloaded_data_path)
        else:
            print("loading data from scratch...")
            train_df = pd.read_csv(args.train_data).sample(frac=args.data_pct, random_state=args.seed)
            print(f"Train data size: {train_df.shape}")
            valid_df = pd.read_csv(args.valid_data)
            print(f"Valid data size: {valid_df.shape}")

            df = pd.concat([train_df, valid_df])

            # load vocab
            print("loading vocab...")
            with open(args.vocab_path, 'rb') as f:
                news_vocab = pickle.load(f)

            print("creating classifier dataloader...")
            old_stdout = sys.stdout  # backup current stdout
            sys.stdout = open(os.devnull, "w")
            classifier_dataloader = DataBlock(blocks=(TextBlock.from_df('text', min_freq=2,
                                              vocab=news_vocab), CategoryBlock),
                                              get_x=ColReader('text'),
                                              get_y=ColReader('label'),
                                              splitter=ColSplitter()).dataloaders(df, bs=args.batch_size, num_workers=0)
            sys.stdout = old_stdout  # reset old stdout

        print("Loading data finished.")
        
        # Initialize model
        print("creating model...")
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        learn = text_classifier_learner(classifier_dataloader, AWD_LSTM, wd=args.weight_decay, metrics=accuracy).to_fp16()
        sys.stdout = old_stdout  # reset old stdout

        # load encoder from pretrained LM
        print(f"loading encoder at {args.pretrained_path}")
        learn.load_encoder(args.pretrained_path)

        print("Model loaded successfully!\n")

        print('\n' + '=' * 50, '\nBEGING TRAINING', '\n' + '=' * 50)
        print('METHOD: Gradual Unfreezing and Discriminative Learning Rates\n\n')
        
        # Set learning rates
        lr = args.lr_max
        print(f"LR_MAX: {lr}")
        # Parse epochs
        epochs = [int(epoch) for epoch in args.epochs.split(";")]

        print(f"EPOCHS: {epochs}")

        # Train the last layers
        with learn.no_bar(), learn.no_mbar():
            print(f"Stage 1:\n Epochs: {epochs[0]}\n lr_max: {lr}")
            learn.fit_one_cycle(epochs[0], lr)

            print(f"Stage 2:\n Epochs: {epochs[1]}\n lr_max: 'slice(lr/(2.6**4), lr)'")
            learn.freeze_to(-2)  
            learn.fit_one_cycle(epochs[1], slice(lr/(2.6**4), lr))

            print(f"Stage 3:\n Epochs: {epochs[2]}\n lr_max: 'slice(lr/2/(2.6**4), lr/2)'")
            learn.freeze_to(-3)  
            learn.fit_one_cycle(epochs[2], slice(lr/2/(2.6**4), lr/2))

            print(f"Stage 3:\n Epochs: {epochs[3]}\n lr_max: 'slice(lr/4/(2.6**4), lr/4)'")
            learn.freeze_to(-4)
            learn.fit_one_cycle(epochs[3], slice(lr/4/(2.6**4), lr/4))

            print(f"Stage 5 (Unfreeze all):\n Epochs: {epochs[4]}\n lr_max: 'slice(lr/10/(2.6**4), lr/10)'")
            learn.unfreeze() 
            learn.fit_one_cycle(epochs[4], slice(lr/10/(2.6**4), lr/10))

        # Save the model as models/args.checkpoint.pkl
        print(f"saving model: {args.checkpoint}")
        learn.save(args.checkpoint)
        print("model saved.")

    if args.do_eval:
        print('\n' + '=' * 50, '\nBEGIN EVALUATION PROPER', '\n' + '=' * 50)

        val_loss, val_acc = evaluate(args, args.valid_data)

        print("Valid Loss {:.4f} | Valid Accuracy {:.4f}".format(
            val_loss, val_acc))

        test_loss, test_acc = evaluate(args, args.test_data)

        print("Test Loss {:.4f} | Test Accuracy {:.4f}".format(
            test_loss, test_acc))

        print('\n' + '=' * 50, '\nRESULTS SUMMARY', '\n' + '=' * 50)
        print("Valid Loss {:.4f} | Valid Accuracy {:.4f}".format(
            val_loss, val_acc))
        print("Test Loss {:.4f} | Test Accuracy {:.4f}".format(
            test_loss, test_acc))

        
        



