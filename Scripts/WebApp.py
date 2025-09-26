import os
import numpy as np
import streamlit as st
import faiss
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Initialiser le client Hugging Face
client = InferenceClient(token=HF_API_KEY)

# Nom du modèle d'embedding
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Charger l’index FAISS (assume que index.faiss + docs.npy existent)
index = faiss.read_index("Scripts/docs.index")
docs = np.load("Scripts/docs.npy", allow_pickle=True)

# --- Fonctions ---

def embed_query(query: str):
    """
    Crée un embedding via Hugging Face Inference API (feature_extraction).
    Retourne un np.array float32 2D (compatible avec FAISS).
    """
    resp = client.feature_extraction(model=embedding_model, inputs=query)  # ✅ correct
    emb_array = np.array(resp, dtype="float32").reshape(1, -1)

    # Debug (affiche la forme de l’embedding pour vérifier)
    st.write("📐 Embedding shape :", emb_array.shape)

    return emb_array


def retrieve_context(query, k=4):
    """
    Récupère les k documents les plus proches dans l’index FAISS.
    """
    qv = embed_query(query)
    D, I = index.search(qv, k=k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append((float(dist), docs[idx]))

    return results


# --- Interface Streamlit ---
st.set_page_config(page_title="EcoFrance - RAG", page_icon="🌍", layout="wide")
st.title("🌍 EcoFrance - Recherche intelligente avec RAG")

question = st.text_input("❓ Pose ta question :", placeholder="Ex: Quels sont les objectifs de l'accord de Paris ?")
k = st.slider("📊 Nombre de documents récupérés", 1, 10, 4)

submit = st.button("🔎 Rechercher")

if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        # Récupérer le contexte via FAISS
        retrieved = retrieve_context(question, k=k)

        # Afficher le contexte utilisé
        st.subheader("📚 Contexte utilisé")
        for dist, doc in retrieved:
            st.markdown(f"- **(score {dist:.4f})** {doc}")
