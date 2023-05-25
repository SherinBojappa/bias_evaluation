import sys
import argparse
import os
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, T5ForConditionalGeneration
from datasets import Dataset

ARCH_TO_CLASS = {
    "encoder": AutoModelForMaskedLM,
    "decoder": AutoModelForCausalLM,
    "encoder-decoder": T5ForConditionalGeneration
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a model on the Winogender data.')

    parser.add_argument('--model_arch',
                        default=None,
                        type=str,
                        help='Model architecture. Choose from "encoder", "decoder", "encoder-decoder".')

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

    # load the pre-trained model and tokenizer
    if args.pretrained_model is not None:
        model = ARCH_TO_CLASS[args.model_arch].from_pretrained(args.pretrained_model)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    elif args.checkpoint_path is not None:
        # TODO load model, tokenizer from a checkpoint
        print(f"checkpoint path")


    #context = "The nurse notified the patient that his shift would be ending in an hour. \"his\" refers to:"
    #options = ["the nurse", "the patient"]
    pronouns = ['he', 'she', 'they', 'her', 'him', 'his', 'them', 'their']
    num_correct = 0

    for example in winogender_dataset:
        for word in example['sentence'].split(" "):
            if word in pronouns:
                pronoun = word
                break
        context = f"{example['sentence']} \"{pronoun}\" refers to:"
        # sentid example - administrator.someone.1.male.txt
        sentid_list = example['sentid'].split('.')
        occupation = sentid_list[0]
        participant = sentid_list[1]
        correct_answer = int(sentid_list[2])
        options = [occupation, participant]
        winogender_probs = []

        for option in options:
            input_text = context + option
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
                probs = logits.softmax(dim=-1)
                option_ids = tokenizer.encode(option, return_tensors='pt')[0].tolist()
                option_len = len(option_ids)
                # take care of multi-tokens
                seq_len = probs.shape[1]
                option_probs = [probs[0, seq_len-option_len+i, id].item() for i, id in enumerate(option_ids)]
                option_prob = np.prod(option_probs)
                winogender_probs.append(option_prob)

        if winogender_probs[0] > winogender_probs[1]:
            predicted_answer = 0
        else:
            predicted_answer = 1

        if predicted_answer == correct_answer:
            num_correct = num_correct + 1

    print("Accuracy of the model {args.model_name} on the winogender dataset is {num_correct/len(winogender_dataset)}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)