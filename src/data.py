"""Caricamento, pulizia e split del dataset di ticket."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def clean_text(s: str) -> str:
    # nel CSV gli a-capo sono salvati come sequenze letterali "\n"
    return str(s).replace("\\n", "\n").strip()


def load_tickets() -> pd.DataFrame:
    """Ticket in inglese, ripuliti, con un campo `text` = oggetto + corpo."""
    df = pd.read_csv(config.RAW_CSV)
    df = df[df["language"] == "en"].copy()
    df = df.dropna(subset=["body", "answer"])
    df["subject"] = df["subject"].fillna("").map(clean_text)
    df["body"] = df["body"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)
    df["text"] = (df["subject"] + "\n\n" + df["body"]).str.strip()
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["ticket_id"] = ["T" + str(i).zfill(5) for i in range(len(df))]
    cols = ["ticket_id", "subject", "body", "text", "answer", "queue", "type", "priority"]
    return df[cols]


def make_split():
    """Split stratificato per coda, salvato su disco per riproducibilita'."""
    df = load_tickets()
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["queue"], random_state=config.RANDOM_STATE
    )
    train.to_parquet(config.TRAIN_FILE, index=False)
    test.to_parquet(config.TEST_FILE, index=False)
    return train, test


def load_split():
    if not config.TRAIN_FILE.exists():
        return make_split()
    return pd.read_parquet(config.TRAIN_FILE), pd.read_parquet(config.TEST_FILE)


if __name__ == "__main__":
    tr, te = make_split()
    print(f"train={len(tr)}  test={len(te)}")
