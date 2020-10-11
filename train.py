import argparse
from utils.training import run_finetuning


def main():
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--pretrained_path', type=str,
                        help='Pre-trained fine-tuned language model located at models/<filename>.pth')
    parser.add_argument('--vocab_path', type=str,
                        help='Path should be models/<filename>.pkl')
    parser.add_argument('--classifier_path', type=str,
                        help='<filename>.pth of the classifier model (for evaluation).')
    parser.add_argument('--checkpoint', type=str,
                        help='Name of output model/checkpoint to evaluate on test data.')
    parser.add_argument('--preloaded_data_path', type=str,
                        help='Path to preloaded dataloader.')
    parser.add_argument('--train_data', type=str,
                        help='Path to the training data.')
    parser.add_argument('--valid_data', type=str,
                        help='Path to the validation data.')
    parser.add_argument('--test_data', type=str,
                        help='Path to the testing data.')
    parser.add_argument('--data_pct', type=float, default=1.0,
                        help='Percentage of training data to train on. Reduce to simulate low-resource settings.')

    # Training parameters
    parser.add_argument('--do_train', action="store_true", 
                        help='Finetune the model.')
    parser.add_argument('--do_eval', action="store_true",
                        help='Evaluate the model.')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batch size.')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='Weight decay.')
    parser.add_argument('--lr_max', type=float,
                        default=0.0, help='Max learning rate.')
    parser.add_argument('--epochs', type=str, default='4;2;2;2;1',
                        help='Number of epochs to train for 5 stages of gradual unfreezing. Must pass 5 values separated by ";". (e.g. 4;2;2;2;1, Stage 1 trains for 4 epochs. Stage 5 trains for 1 epoch.')
    parser.add_argument('--seed', type=int, default=42, help='Set random seed.')


    args = parser.parse_args()

    # Log the configuration
    print('=' * 50, '\nCONFIGURATION', '\n' + '=' * 50)
    for key in vars(args):
        print("{}: {}".format(key, vars(args)[key]))


    # Run finetuning
    print('\n' + '=' * 50, '\nEXECUTE RUN_FINETUNING()', '\n' + '=' * 50)
    run_finetuning(args)


if __name__ == '__main__':
    main()
