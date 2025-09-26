#### WebApp.py (Chatbot RAG avec embeddings locaux)
###### Prérequis :
# - docs.index et docs.json créés par build_index.py
# - Installer la librairie : pip install sentence-transformers faiss-cpu streamlit numpy

import streamlit as st
import faiss
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index_file = os.path.join(BASE_DIR, "docs.index")
docs_file = os.path.join(BASE_DIR, "docs.json")

# Modèles
embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"

k = 4

# === Initialisation du modèle d'embedding local ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer(embedding_model_name)

embedder = load_embedder()

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
    Crée un embedding via SentenceTransformer local et renvoie un np.array float32 2D.
    Compatible avec Faiss.
    """
    emb_array = embedder.encode([query], convert_to_numpy=True)
    return emb_array.astype('float32')

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
    prompt = (
        f"{system}\n\n"
        "Contexte récupéré (extraits pertinents) :\n"
        f"{context}\n\n"
        f"Question : {question}\n\n"
        "Réponds de manière claire et concise en t'appuyant sur le contexte. "
        "Si l'information n'est pas dans le contexte, dis-le et propose comment l'obtenir."
    )
    return prompt

# === LOGIQUE CHAT ===
if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        retrieved = retrieve_context(question, k=k)
        st.subheader("📚 Contexte utilisé")
        for r in retrieved:
            st.markdown(f"- [chunk id={r['id']} | dist={r['distance']:.4f}] {r['text'][:200]}...")

        prompt = build_prompt(question, retrieved)
        st.subheader("💡 Prompt généré")
        st.text_area("Prompt", prompt, height=300)

        # Ici, tu peux ajouter l'appel au modèle LLM via Hugging Face si tu le souhaites
        # par exemple : réponse = client.model(llm_model)(prompt)
        # st.subheader("🤖 Réponse")
        # st.write(réponse)
