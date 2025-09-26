##########
import streamlit as st
import faiss
import numpy as np
import json
from huggingface_hub import InferenceClient

# --- Config ---
HUGGINGFACE_API_KEY = "hf_xxxxx"  # ⚠️ Mets ton vrai token ici ou dans les secrets Streamlit
client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# --- Charger l’index FAISS et les docs ---
index = faiss.read_index("Scripts/docs.index")

with open("Scripts/docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# --- Fonctions ---
def embed_query(query: str):
    """
    Crée un embedding via Hugging Face Inference API (feature_extraction).
    Compatible avec Faiss.
    """
    resp = client.feature_extraction(model=embedding_model, inputs=query)
    emb_array = np.array(resp, dtype="float32").reshape(1, -1)  # 2D pour FAISS
    return emb_array

def retrieve_context(query, k=4):
    qv = embed_query(query)
    D, I = index.search(qv, k=k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < len(docs):  # Sécurité
            results.append((dist, docs[idx]))
    return results

# --- UI ---
st.title("🔎 RAG EcoFrance")

question = st.text_input("Pose ta question :")
k = st.slider("Nombre de documents à récupérer", 1, 10, 4)
submit = st.button("Rechercher")

if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        # Récupérer le contexte via Faiss
        retrieved = retrieve_context(question, k=k)

        # Afficher le contexte utilisé
        st.subheader("📚 Contexte utilisé")
        for dist, doc in retrieved:
            st.write(f"**Score:** {dist:.4f}")
            st.write(doc)
            st.markdown("---")
