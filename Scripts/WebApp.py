import os
import numpy as np
import faiss
import streamlit as st
from huggingface_hub import HfInference
from dotenv import load_dotenv

# =====================
# 🔧 Configuration
# =====================
load_dotenv()
hf_token = os.environ.get("HUGGINGFACE_API_KEY")
embedding_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not hf_token:
    st.error("❌ Clé Hugging Face API manquante. Ajoute-la dans tes secrets ou .env.")
    st.stop()

# Client Hugging Face Inference
client = HfInference(token=hf_token)

# =====================
# 📚 Fonctions
# =====================
def embed_query(query: str):
    """
    Crée un embedding via Hugging Face et renvoie un np.array float32 2D
    Compatible avec Faiss.
    """
    emb = client.embed_text(model=embedding_model, text=query)
    emb_array = np.array(emb, dtype="float32").reshape(1, -1)
    return emb_array

def retrieve_context(query, k=4):
    """
    Récupère les k passages les plus proches avec Faiss.
    """
    qv = embed_query(query)
    D, I = index.search(qv, k=k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx != -1:
            results.append(docs[idx])
    return results

# =====================
# 📖 Chargement index
# =====================
try:
    index = faiss.read_index("faiss_index.bin")
    with open("docs_store.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"❌ Impossible de charger l'index : {e}")
    st.stop()

# =====================
# 🎨 Interface Streamlit
# =====================
st.set_page_config(page_title="RAG EcoFrance", page_icon="🌱", layout="wide")
st.title("🌱 RAG EcoFrance")
st.write("Pose ta question, je vais chercher dans la base et générer une réponse.")

question = st.text_input("❓ Pose ta question :")
k = st.slider("📊 Nombre de passages à récupérer :", 1, 10, 4)
submit = st.button("🔍 Rechercher")

if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        retrieved = retrieve_context(question, k=k)

        st.subheader("📚 Contexte utilisé")
        for r in retrieved:
            st.markdown(f"- {r}")

        st.subheader("🤖 Réponse (placeholder)")
        st.info("👉 Ici tu peux ajouter l’appel à un modèle génératif (Ollama, OpenAI, etc.)")
