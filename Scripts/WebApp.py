## WebApp.py (Chatbot RAG avec Hugging Face APPLICATION)
import streamlit as st
import faiss
import numpy as np
import json
import os
import requests
from dotenv import load_dotenv

# === CHARGER VARIABLES D'ENVIRONNEMENT ===
# Si fichier .env local disponible
env_path = "Scripts/clehug.env"
if os.path.exists(env_path):
    load_dotenv(env_path)

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier du script
index_file = os.path.join(BASE_DIR, "docs.index")
docs_file = os.path.join(BASE_DIR, "docs.json")

# Modèles Hugging Face
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
k = 4

# === Initialisation Hugging Face API ===
hf_token = os.environ.get("HUGGINGFACE_API_KEY")

if not hf_token:
    st.error("❌ Clé Hugging Face API manquante. "
             "Ajoutez-la dans le fichier .env local ou dans les Secrets de Streamlit Cloud.")
    st.stop()
else:
    st.success("✅ Clé Hugging Face chargée correctement")

HEADERS = {"Authorization": f"Bearer {hf_token}"}

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
    """Créer un embedding via Hugging Face API"""
    url = f"https://api-inference.huggingface.co/models/{embedding_model}"
    response = requests.post(url, headers=HEADERS, json={"inputs": query})
    response.raise_for_status()
    return np.array(response.json(), dtype="float32").reshape(1, -1)

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
Si l'information n'est pas dans le contexte, dis-le et propose comment l'obtenir."""
    return prompt

def generate_answer(prompt: str):
    """Générer une réponse via Hugging Face API"""
    url = f"https://api-inference.huggingface.co/models/{llm_model}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    # L'API renvoie une liste avec "generated_text"
    return response.json()[0]["generated_text"]

# === CHAT ===
if submit and question.strip():
    with st.spinner("🔎 Recherche dans l'index et génération de la réponse..."):
        retrieved = retrieve_context(question, k=k)

        st.subheader("📚 Contexte utilisé")
        for r in retrieved:
            with st.expander(f"Chunk {r['id']} (distance={r['distance']:.4f})"):
                st.write(r['text'])

        prompt = build_prompt(question, retrieved)
        answer = generate_answer(prompt)

        # Affichage stylé de la réponse
        st.subheader("🤖 Réponse du modèle")
        st.markdown(
            f"""
            <div style="border:2px solid #4CAF50; padding:15px; border-radius:10px; background-color:#f9fff9;">
                {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )
