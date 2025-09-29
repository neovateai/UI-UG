"""Configuration constants for evaluation scripts."""

import os

# Model configuration
MODEL_PATH = os.getenv('MODEL_PATH')
CUDA_DEVICES = os.getenv('CUDA_DEVICES', '0')

# Image processing configuration
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# Default batch size
DEFAULT_BATCH_SIZE = 1

# Output path configuration
OUTPUT_PATH = 'eval_results'

# Sampling parameters configuration
DEFAULT_SAMPLING_CONFIG = {
    'temperature': 0.0,
    'repetition_penalty': 1.2,
    'max_tokens': 8192,
    'stop_token_ids': [],
    'skip_special_tokens': False
}

# VLLM configuration
VLLM_CONFIG = {
    'limit_mm_per_prompt': {"image": 2, "video": 2},
}