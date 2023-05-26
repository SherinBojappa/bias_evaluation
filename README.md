# bias_evaluation

## Github repository for bias evaluation on standard bias datasets.

#### 1. Original repo for crows-pairs - https://github.com/nyu-mll/crows-pairs/tree/master

#### 2. Original repo for winogender schemas - https://github.com/rudinger/winogender-schemas/tree/master
```bash
python3 evaluate.py \
  --model_arch decoder \
  --pretrained_model gpt2 \
  --dataset_path data/all_sentences.tsv \
  --output_file gpt2_winogender.csv

