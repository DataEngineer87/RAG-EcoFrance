# 🤖 Chatbot RAG — Économie Française  

![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-000000?logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0055A4?logo=apache&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


Un assistant IA capable d’analyser et d’interroger un rapport PDF en langage naturel.
Grâce à une architecture RAG (Retrieval-Augmented Generation), le chatbot combine recherche d’information et génération de texte pour fournir des réponses précises et contextualisées.

## Points clés
- Ollama (Mistral) → Modèle de langage open-source pour embeddings & génération
- FAISS → Recherche vectorielle rapide et efficace
- Streamlit → Interface web interactive et simple à utiliser

## Fonctionnalités
- Extraction et découpage automatique d’un PDF
- Création d’embeddings avec mistral
- Indexation avec FAISS
- Recherche vectorielle des chunks pertinents
- Génération de réponse contextualisée avec un LLM
- Interface web interactive avec Streamlit

## Architecture du pipeline
Organigramme TD
   - A[PDF Rapport] --> B[Extraction Texte]
   - B --> C[Découpage en Chunks]
   -  C --> D[Embeddings (Ollama)]
   -  D --> E[FAISS Index]
   -  F[Question Utilisateur] --> G[Embedding Question]
   -  G --> H[Recherche FAISS]
   - H --> I[Contexte Pertinent]
   - I --> J[LLM (Mistral via Ollama)]
   - J --> K[Réponse Contextualisée]
   - K --> L[Interface Streamlit]
### Schéma
![Pipeline RAG](images/Pipeline_rag_chatbot_dark.png)

## Installation et exécution
### Prérequis :
- Ollama installé et lancé
- Téléchargement du modèle :

----
ollama pull mistral
----

----
ollama serve
----
