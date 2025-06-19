import os
from dotenv import load_dotenv, dotenv_values

from src.utils.logging import get_logger

logger = get_logger()
load_dotenv(".env")


class Config:
    HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE")
    if not HF_DATASETS_CACHE:
        raise ValueError("HF_DATASETS_CACHE is not configured")
    
    BASELINE_DATASET_PATH = os.getenv("BASELINE_DATASET_PATH")
    if not BASELINE_DATASET_PATH:
        raise ValueError("BASELINE_DATASET_PATH is not configured")
    
    BASELINE_OUT_DIR = os.getenv("BASELINE_OUT_DIR")
    if not BASELINE_OUT_DIR:
        raise ValueError("BASELINE_OUT_DIR is not configured")
    
    BASELINE_METRICS_DIR = os.getenv("BASELINE_METRICS_DIR")
    if not BASELINE_METRICS_DIR:
        raise ValueError("BASELINE_METRICS_DIR is not configured")
    

    IMPROVED_OUT_DIR = os.getenv("IMPROVED_OUT_DIR")
    if not IMPROVED_OUT_DIR:
        raise ValueError("IMPROVED_OUT_DIR is not configured")
    
    IMPROVED_METRICS_DIR = os.getenv("IMPROVED_METRICS_DIR")
    if not IMPROVED_METRICS_DIR:
        raise ValueError("IMPROVED_METRICS_DIR is not configured")
    
