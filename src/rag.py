"""Knowledge base dei ticket risolti: embedding, retrieval e bozza di risposta (RAG)."""
import numpy as np
import pandas as pd
from openai import OpenAI

from src import config
from src.guardrails import mask_pii

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # legge OPENAI_API_KEY dall'ambiente / .env
    return _client


def embed(texts, batch_size=256) -> np.ndarray:
    """Embedding normalizzati (norma 1), cosi' il prodotto scalare e' la similarita' coseno."""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i:i + batch_size]]
        resp = client().embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
        print(f"  embedding {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index(train: pd.DataFrame):
    """Indicizza i ticket risolti del training set (testo del ticket -> risposta dell'operatore)."""
    kb = train[["ticket_id", "subject", "text", "answer", "queue"]].reset_index(drop=True)
    emb = embed([mask_pii(t) for t in kb["text"].tolist()])
    np.save(config.INDEX_EMB_FILE, emb)
    kb.to_parquet(config.INDEX_META_FILE, index=False)
    print(f"\nKnowledge base creata: {len(kb)} ticket")


class KnowledgeBase:
    def __init__(self):
        self.emb = np.load(config.INDEX_EMB_FILE)
        self.meta = pd.read_parquet(config.INDEX_META_FILE)

    def search(self, query_vec: np.ndarray, k=config.TOP_K, queue=None) -> pd.DataFrame:
        scores = self.emb @ query_vec
        if queue is not None:
            scores = np.where(self.meta["queue"].to_numpy() == queue, scores, -1.0)
        top = np.argsort(-scores)[:k]
        hits = self.meta.iloc[top].copy()
        hits["similarity"] = scores[top]
        return hits.reset_index(drop=True)

    def retrieve(self, text: str, k=config.TOP_K, queue=None) -> pd.DataFrame:
        return self.search(embed([mask_pii(text)])[0], k=k, queue=queue)


SYSTEM_PROMPT = """You are Helpdesk Copilot, an assistant that drafts replies for human support agents.
Rules (must always be followed):
- Use ONLY the information in the RESOLVED TICKETS provided. Do not invent policies, prices, deadlines, links or technical facts.
- If the resolved tickets are not sufficient to answer, reply exactly: INSUFFICIENT_CONTEXT
- Never ask for passwords, card numbers or other sensitive data.
- Keep placeholders like <EMAIL> or <name> as they are.
- Write a polite, concise reply in English addressed to the customer (max 150 words).
- After the reply, add a line "Sources: " followed by the IDs of the tickets you used."""


def draft_reply(ticket_text: str, hits: pd.DataFrame) -> dict:
    """Genera una bozza solo se il retrieval e' abbastanza affidabile (guardrail)."""
    if hits.empty or hits["similarity"].max() < config.SIMILARITY_THRESHOLD:
        return {"status": "no_context",
                "draft": "Nessun ticket storico sufficientemente simile: serve l'intervento di un operatore."}

    context = "\n\n".join(
        f"[{r.ticket_id}] (queue: {r.queue}, similarity: {r.similarity:.2f})\n"
        f"CUSTOMER: {r.text}\nAGENT REPLY: {r.answer}"
        for r in hits.itertuples()
    )
    resp = client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"RESOLVED TICKETS:\n{mask_pii(context)}\n\n"
                                        f"NEW TICKET:\n{mask_pii(ticket_text)}"},
        ],
    )
    draft = resp.choices[0].message.content.strip()
    if "INSUFFICIENT_CONTEXT" in draft:
        return {"status": "no_context",
                "draft": "Il modello ha giudicato insufficienti i ticket storici: serve l'intervento di un operatore."}
    return {"status": "ok", "draft": draft}


if __name__ == "__main__":
    from src.data import load_split
    train, _ = load_split()
    build_index(train)
