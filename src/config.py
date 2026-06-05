
"""
config.py — Central configuration for all pipeline stages.
Edit this file to change model, data, training, or evaluation settings.
"""

# Model
MODEL_NAME   = 'distilbert-base-cased'
# Choose device: 'mps' for Apple Silicon, 'cuda' for NVIDIA GPU, 'cpu' for local fallback.
DEVICE_NAME={
    'mps': 'mps',
    'cuda': 'cuda',
    'cpu': 'cpu',
}

# = 'mps'
# # DEVICE_NAME  = 'cuda'
# # DEVICE_NAME  = 'cpu' 

# File paths
ARTIFACTS_DIR       = './artifacts/'
RAW_DATA_FILE       = ARTIFACTS_DIR+'raw_data_dict.pickle'
PROCESSED_DATA_FILE = ARTIFACTS_DIR+'processed_data_dict.pickle'
ID2LABEL_FILE       = ARTIFACTS_DIR+'id2label.json'
CACHED_MODEL_DIR    = ARTIFACTS_DIR+'model_cache'
# EVAL_OUTPUT_DIR     = './eval_results'
EVAL_OUTPUT_DIR     = ARTIFACTS_DIR+'eval_results'  # Save eval results inside model cache for easier access



# Data loading and sampling settings
MAX_LENGTH        = 512    # maximum token length for DistilBERT
HEAD              = 10000  # upper bound per genre stream before sampling
SAMPLE_SIZE       = 200   # random sample per genre after reading
REVIEWS_PER_GENRE = 1000   # final reviews per genre for train+test split
TRAIN_RATIO       = 0.8

GENRE_URL_DICT = {
    'poetry':                 
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    
    'children':               
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    
    'comics_graphic':
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    
    'fantasy_paranormal':
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    
    'history_biography':
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    
    'mystery_thriller_crime': 
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    
    'romance':                
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    
    'young_adult':            
        'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz',
}


# Weights & Biases

WANDB_PROJECT  = 'mlops-group23-project'
WANDB_RUN_NAME = 'run-1'


# Training

TRAINING_ARGS = dict(
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    report_to='wandb',
    run_name=WANDB_RUN_NAME,
)
