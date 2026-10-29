"""Helpdesk Copilot - demo Streamlit.  Avvio:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from src import config
from src.data import load_split
from src.guardrails import validate_ticket
from src.rag import KnowledgeBase, draft_reply

st.set_page_config(page_title="Helpdesk Copilot", page_icon="🎧", layout="wide")


@st.cache_resource
def load_resources():
    return joblib.load(config.CLASSIFIER_FILE), KnowledgeBase(), load_split()[1]


model, kb, test = load_resources()

st.title("🎧 Helpdesk Copilot")
st.caption("Smistamento automatico dei ticket + bozza di risposta basata sullo storico (RAG). "
           "L'operatore decide sempre.")

with st.sidebar:
    st.header("Ticket di esempio")
    st.write("Ticket del test set, mai visti dal modello.")
    if st.button("🎲 Carica un ticket casuale"):
        row = test.sample(1).iloc[0]
        st.session_state["ticket"] = row["text"]
        st.session_state["true_queue"] = row["queue"]
    st.divider()
    st.header("Guardrail attivi")
    st.markdown(f"""
- Confidenza minima per lo smistamento automatico: **{config.CONFIDENCE_THRESHOLD:.0%}**
- Similarità minima per generare una bozza: **{config.SIMILARITY_THRESHOLD}**
- Mascheramento dati personali prima dell'invio all'API
- L'LLM risponde solo sulla base dei ticket recuperati
""")

ticket = st.text_area("Testo del ticket (inglese)", key="ticket", height=200)
if "true_queue" in st.session_state:
    st.caption(f"Coda reale (solo per la demo): {st.session_state['true_queue']}")

if st.button("Analizza ticket", type="primary"):
    ok, msg = validate_ticket(ticket)
    if not ok:
        st.error(msg)
        st.stop()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1 · Smistamento")
        proba = pd.Series(model.predict_proba([ticket])[0], index=model.classes_).sort_values(ascending=False)
        queue, conf = proba.index[0], proba.iloc[0]
        if conf >= config.CONFIDENCE_THRESHOLD:
            st.success(f"**{queue}**  \nconfidenza {conf:.0%}: smistamento automatico")
        else:
            st.warning(f"Suggerita: **{queue}**  \nconfidenza {conf:.0%}: **revisione umana**")
        st.bar_chart(proba.head(5))

    with col2:
        st.subheader("2 · Bozza di risposta")
        with st.spinner("Cerco ticket simili e preparo la bozza..."):
            hits = kb.retrieve(ticket)
            out = draft_reply(ticket, hits)
        if out["status"] == "ok":
            st.text_area("Bozza (modificabile dall'operatore)", out["draft"], height=260)
        else:
            st.warning(out["draft"])

        st.subheader("3 · Ticket storici usati come fonte")
        for h in hits.itertuples():
            with st.expander(f"{h.ticket_id} · {h.queue} · similarità {h.similarity:.2f}"):
                st.markdown(f"**Cliente:** {h.text}")
                st.markdown(f"**Risposta operatore:** {h.answer}")
