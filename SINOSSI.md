# Helpdesk Copilot
### Smistamento automatico dei ticket e riuso della conoscenza con Machine Learning e RAG

**Capstone Project – Master in Python, AI & Machine Learning (EPICODE)**
**Autore:** Giovanni

---

## 1. Problema
Un helpdesk aziendale riceve ogni giorno centinaia di ticket. Due attività consumano gran parte del tempo degli operatori:

1. **Smistamento (triage):** leggere ogni ticket e assegnarlo alla coda corretta (supporto tecnico, fatturazione, IT, resi, …). Un errore di smistamento allunga i tempi di risoluzione perché il ticket rimbalza tra i team.
2. **Risposta:** molte richieste sono simili a ticket già risolti, ma la conoscenza contenuta nelle risposte passate non viene riusata: l'operatore riscrive ogni volta da zero.

È un tipico problema di **knowledge management**: l'azienda possiede già la conoscenza (lo storico dei ticket risolti), ma non è organizzata né recuperabile nel momento in cui serve.

## 2. Obiettivo
Realizzare un prototipo, **Helpdesk Copilot**, che affianca l'operatore (*human-in-the-loop*, senza sostituirlo) e:

- **classifica automaticamente** un nuovo ticket nella coda corretta, indicando un livello di confidenza;
- **recupera i ticket storici più simili** e genera una **bozza di risposta** basata esclusivamente su di essi, citando le fonti.

## 3. Perimetro (cosa è incluso e cosa no)
| Incluso | Escluso |
|---|---|
| Ticket in lingua inglese | Altre lingue |
| Le 10 code principali dell'helpdesk | Integrazione con sistemi di ticketing reali (Jira, Zendesk, …) |
| Classificazione della coda + bozza di risposta | Invio automatico delle risposte al cliente |
| Prototipo dimostrativo (app web locale) | Messa in produzione, autenticazione, scalabilità |

La decisione finale resta **sempre all'operatore**.

## 4. Dataset
**Customer Support Tickets** (Hugging Face, *Tobi-Bueck/customer-support-tickets*, licenza CC BY-NC 4.0).
- 28.587 ticket, di cui **16.338 in inglese** (sottoinsieme utilizzato);
- campi: oggetto, testo, **risposta dell'operatore**, **coda** (10 classi), tipo, priorità, tag;
- classi sbilanciate (da circa 4.700 ticket per *Technical Support* a circa 240 per *General Inquiry*).

Limite dichiarato: il dataset è di origine semi-sintetica. È adatto alla prototipazione, ma i risultati andrebbero validati su dati reali aziendali.

## 5. Metodologia
1. **Analisi esplorativa (EDA):** distribuzione delle classi, lunghezza dei testi, valori mancanti, parole più frequenti per coda.
2. **Preprocessing del testo:** unione di oggetto e corpo, pulizia, vettorizzazione **TF-IDF**.
3. **Classificazione della coda (Machine Learning):** confronto tra *Logistic Regression*, *Linear SVM*, *Naive Bayes* e una **rete neurale (MLP)**, con split stratificato train/test. Metriche: accuracy, F1 macro, matrice di confusione.
4. **Knowledge base (RAG):** i ticket risolti del training set vengono convertiti in **embedding** (OpenAI) e indicizzati. Per un nuovo ticket si recuperano i *k* più simili tramite similarità coseno.
5. **Generazione della bozza (GenAI):** un LLM (OpenAI) redige la risposta usando solo i ticket recuperati.
6. **Valutazione del retrieval:** percentuale di casi in cui i ticket recuperati appartengono alla stessa coda del ticket di test (*hit-rate@k*).
7. **Demo:** applicazione **Streamlit** in cui l'operatore incolla un ticket e ottiene coda, confidenza, ticket simili e bozza di risposta.

## 6. Guardrail
- **Soglia di confidenza:** sotto una soglia il ticket non viene smistato automaticamente ma inviato a revisione umana.
- **Risposte vincolate (grounding):** l'LLM risponde solo sulla base dei ticket recuperati e cita le fonti. Se la somiglianza è troppo bassa, il sistema dichiara di non avere una risposta affidabile.
- **Privacy:** email, numeri di telefono e altri dati personali vengono mascherati prima dell'invio all'API esterna.
- **Human-in-the-loop:** il sistema produce solo suggerimenti, nessuna azione automatica verso il cliente.

## 7. Tecnologie
Python 3.12 · Pandas · NumPy · Matplotlib/Seaborn · scikit-learn · OpenAI API (embedding + LLM) · Streamlit · Jupyter

## 8. Moduli del Master applicati
| Modulo | Applicazione nel progetto |
|---|---|
| Python | Struttura modulare del codice, script riutilizzabili |
| Python for Data Science | Pulizia dei dati ed EDA con Pandas e Seaborn |
| Machine Learning | Classificazione supervisionata, confronto tra modelli, metriche |
| Deep Learning | Rete neurale MLP confrontata con i modelli classici |
| NLP | Preprocessing del testo, TF-IDF, embedding semantici |
| Generative AI | Pipeline RAG con LLM, prompt con guardrail, app finale |

## 9. Risultati attesi
- Un classificatore di coda con prestazioni nettamente superiori alla baseline (classe più frequente) e una soglia di confidenza per i casi incerti;
- un sistema di recupero e generazione che propone bozze coerenti con lo storico;
- un'app dimostrativa utilizzabile da un operatore.

## 10. Limiti e sviluppi futuri
- Dataset semi-sintetico: serve una validazione su ticket reali;
- modelli Transformer *fine-tuned* (es. BERT) per migliorare la classificazione;
- supporto multilingua (il dataset contiene anche ticket in tedesco);
- feedback dell'operatore sulle bozze per migliorare il sistema nel tempo;
- integrazione con un sistema di ticketing reale.
