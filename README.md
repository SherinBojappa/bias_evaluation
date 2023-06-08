# bias_evaluation

The following instructions will guide you through running evaluations on the Crows-Pairs and Winogender Schemas datasets.

```bash
# Navigate to the crows-pairs directory and run the evaluation
cd crows-pairs
python3 metric.py \
  --input_file data/crows_pairs_anonymized.csv \
  --lm_model bert \
  --output_file bert_crowspairs

# Navigate to the winogender directory and run its evaluation
cd winogender
python3 evaluate.py \
  --model_arch decoder \
  --pretrained_model gpt2 \
  --dataset_path data/all_sentences.tsv \
  --chatbot_model

# To optionally specify a checkpoint model path you can use the following command
# where ACTOR_MODEL_PATH is the environment variable contatining the path of the model.
cd winogender
python3 evaluate.py \
  --model_arch decoder \
  --checkpoint_path $ACTOR_MODEL_PATH \
  --dataset_path data/all_sentences.tsv \
  --chatbot_model
