import sys
import argparse
import os

from datasets import Dataset

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a model on the Winogender data.')

    parser.add_argument('--pretrained_model',
                        default=None,
                        type=str,
                        help='Load a pre-trained model from HuggingFace. Provide the model name as the argument.')

    parser.add_argument('--checkpoint_path',
                        default=None,
                        type=str,
                        help='Load a model from the specified checkpoint. Provide the path to the checkpoint as the argument.')

    parser.add_argument('--dataset_path',
                        required=True,
                        type=str,
                        help='Path containing the data. This argument is required.')

    parser.add_argument('--output_file',
                        default='output.txt',
                        type=str,
                        help='Name of the file in which the output needs to be stored. Defaults to "output.txt".')

    return parser.parse_args()


def main(args):
    # load the windogender data
    # row 0 - sentid	sentence
    # row 1 - row 720 - actual sentences
    with open(args.dataset_path, 'r') as f:
        winogender_data = [line.strip().split('\t') for line in f.readlines()]

    # skip row 0 as it has sendid and sentence header and not actual data
    data_dict = {
        "sentid": [line[0] for line in winogender_data[1:]],
        "sentence": [line[1] for line in winogender_data[1:]],
        }
    winogender_dataset = Dataset.from_dict(data_dict)

    # load the pre-trained model


    # obtain predictions

    # compute accuracy

    # analyse bias in predictions

    # save the results


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
