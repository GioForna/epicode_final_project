"""Scarica il dataset da Hugging Face in data/tickets_raw.csv."""
import shutil

from huggingface_hub import hf_hub_download

from src import config

REPO = "Tobi-Bueck/customer-support-tickets"
FILE = "aa_dataset-tickets-multi-lang-5-2-50-version.csv"

if __name__ == "__main__":
    path = hf_hub_download(REPO, FILE, repo_type="dataset")
    shutil.copy(path, config.RAW_CSV)
    print("Salvato in", config.RAW_CSV)
