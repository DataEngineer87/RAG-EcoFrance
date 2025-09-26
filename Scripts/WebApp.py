# WebApp.py (Chatbot RAG avec Hugging Face API)
# Prérequis :
# - Scripts/docs.index et Scripts/docs.json créés par build_index.py
# - Hugging Face API key ajoutée dans Streamlit Cloud (Secrets : HUGGINGFACE_API_KEY)

import streamlit as st
import faiss
import numpy as np
import json
import os
from huggingface_hub import InferenceClient

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_file = os.path.join(BASE_DIR, "docs.index")
docs_file = os.path.join(BASE_DIR, "docs.json")

# Modèles Hugging Face
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"

k = 4

# === Initialisation Hugging Face API ===
hf_token = os.environ.get("HUGGINGFACE_API_KEY")
if not hf_token:
    st.error("❌ Clé Hugging Face API manquante. Ajoutez-la dans les Secrets de Streamlit Cloud.")
    st.stop()

client = InferenceClient(token=hf_token)

# === Charger index et documents ===
@st.cache_resource
def load_index_and_docs():
    if not os.path.exists(index_file):
        st.error(f"❌ Fichier introuvable : {index_file}")
        st.stop()
    if not os.path.exists(docs_file):
        st.error(f"❌ Fichier introuvable : {docs_file}")
        st.stop()

    index = faiss.read_index(index_file)
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs

index, docs = load_index_and_docs()

# === UI PRINCIPALE ===
st.set_page_config(page_title="Chatbot RAG", page_icon="🤖", layout="centered")

st.markdown(
    "<h3 style='text-align: center;'>🤖 CHATBOT RAG - INFORMATION SUR L'ÉCONOMIE FRANÇAISE</h3>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'>Que souhaitez-vous savoir sur l'économie française ?</h4>",
    unsafe_allow_html=True
)
st.divider()

question = st.text_input(
    "💬 Entrez votre question :", 
    placeholder="Ex: Quels sont les facteurs influençant le chômage ?"
)
submit = st.button("🚀 Envoyer")

# === FONCTIONS ===
def embed_query(query: str):
    """
    Crée un embedding via Hugging Face Inference API (feature_extraction).
    Compatible avec Faiss.
    """
    # ✅ Correction : utilisation de 'text' au lieu de 'inputs'
    resp = client.feature_extraction(model=embedding_model, text=query)
    emb_array = np.array(resp, dtype="float32").reshape(1, -1)  # 2D pour Faiss
    return emb_array

def retrieve_context(query, k=4):
    qv = embed_query(query)
    D, I = index.search(qv, k=k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        text = docs[idx]
        results.append({"id": int(idx), "distance": float(dist), "text": text})
    return results

def build_prompt(question, retrieved):
    system = "Tu es un assistant expert en économie française. Réponds en français, cite les sections utilisées si utile."
    context = "\n\n---\n".join(
        [f"[chunk id={r['id']} | dist={r['distance']:.4f}]\n{r['text']}" for r in retrieved]
    )
    prompt = f"""{system}

Contexte récupéré (extraits pertinents) :
{context}

Question : {question}

Réponds de manière claire et concise en t'appuyant sur le contexte. 
Si l'information n'est pas dans le contexte, dis-le et propose comment l'obtenir.
"""
    return prompt

def generate_answer(prompt):
    resp = client.text_generation(model=llm_model, inputs=prompt, max_new_tokens=512)
    return resp[0]['generated_text'] if resp else "❌ Pas de réponse générée."

# === LOGIQUE PRINCIPALE ===
if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        # Récupérer le contexte via Faiss
        retrieved = retrieve_context(question, k=k)
        
        # Afficher le contexte utilisé
        st.subheader("📚 Contexte utilisé")
        for r in retrieved:
            st.markdown(f"[chunk id={r['id']} | dist={r['distance']:.4f}]\n{r['text']}")

        # Générer la réponse
        prompt = build_prompt(question, retrieved)
        answer = generate_answer(prompt)
        st.subheader("💡 Réponse générée")
        st.write(answer)
