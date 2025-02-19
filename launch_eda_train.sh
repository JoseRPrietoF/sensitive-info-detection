jupyter nbconvert --to script notebooks/EDA.ipynb
cd notebooks
python -m EDA.py
cd ..

# Train with TinyBERT (fast and lightweight)
python -m src.train \
    --experiment_name "experiment_1" \
    --train_path "data/train.csv" \
    --test_path "data/validation.csv" \
    --model_name prajjwal1/bert-tiny \
    --num_frozen_layers 0 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 2e-4 \
    --early_stopping_patience 10

# Train with ModernBERT (better performance)
python -m src.train \
    --experiment_name "experiment_2" \
    --train_path "data/train.csv" \
    --test_path "data/validation.csv" \
    --model_name answerdotai/ModernBERT-base \
    --num_frozen_layers 0 \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 1e-5 \
    --early_stopping_patience 10
