# bias_evaluation

## Github repository for bias evaluation on standard bias datasets.

#### 1. Original repo for crows-pairs - https://github.com/nyu-mll/crows-pairs/tree/master
```bash
cd crows-pairs
python3 metric.py --input_file data/crows_pairs_anonymized.csv --lm_model bert --output_file bert_crowspairs

#### 2. Original repo for winogender schemas - https://github.com/rudinger/winogender-schemas/tree/master
```bash
cd winogender
python3 evaluate.py \
  --model_arch decoder \
  --pretrained_model gpt2 \
  --dataset_path data/all_sentences.tsv \
  --output_file gpt2_winogender.csv

