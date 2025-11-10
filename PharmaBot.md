---

\# Pharmacy Chatbot using Retrieval-Augmented Generation (RAG)

\>  An AI-powered pharmacy chatbot that retrieves and explains pharmaceutical terminology using \*\*RAG (Retrieval-Augmented Generation)\*\*, \*\*FAISS\*\*, and \*\*Intent Classification\*\* — ensuring safe and accurate responses without providing prescription or dosage advice.

\!\[Python\](https://img.shields.io/badge/Python-3.10-blue)  
\!\[Gradio\](https://img.shields.io/badge/UI-Gradio-orange)  
\!\[FAISS\](https://img.shields.io/badge/Search-FAISS-green)  
\!\[OpenAI\](https://img.shields.io/badge/LLM-OpenAI%20GPT--4o--mini-lightgrey)  
\!\[License\](https://img.shields.io/badge/License-MIT-blue)

\---

\#\#  Project Overview

\*\*PharmaBot\*\* is a \*\*domain-specific AI chatbot\*\* built to understand and respond to pharmaceutical questions ethically and accurately.    
It uses \*\*Retrieval-Augmented Generation (RAG)\*\* — combining \*\*semantic search\*\* with \*\*generative AI\*\* — to deliver context-rich responses from a verified data source:    
 \*WHO Collaborating Centre for Pharmaceutical Pricing and Reimbursement Policies Glossary (2016).\*

The bot automatically refuses unsafe questions (like medicine dosages) and provides information suitable for students, pharmacists, and researchers.

\---

\#\#  Tech Stack

| Layer | Library / Tool | Purpose |  
|-------|----------------|----------|  
|  NLP Model | SentenceTransformer (\`all-MiniLM-L6-v2\`) | Create embeddings for semantic search |  
|  Vector Search | FAISS | Fast similarity-based retrieval |  
|  Re-ranking | CrossEncoder (\`ms-marco-MiniLM-L-6-v2\`) | Improve ranking of search results |  
|  Interface | Gradio | Web-based chatbot UI |  
|  ML Classifier | Scikit-learn (TF-IDF \+ Logistic Regression) | Detect unsafe queries |  
|  Text Parsing | PyPDF2 | Extract text from Pharmacy PDF |  
|  Optional LLM | OpenAI GPT-4o-mini | Generate natural explanations (RAG mode) |  
|  Environment | Google Colab / Jupyter Notebook | Development environment |  
|  Storage | Google Drive | Store preprocessed data and models |

\---

\#\#  Architecture

 Pharmacy Dictionary (PDF)  
 │  
 ▼  
  Text Cleaning →  Chunking →  Embeddings  
 │ │  
 ▼ ▼  
 FAISS Vector DB Intent Classifier  
 │ │  
 └──\> Query → Retrieve → Re-rank → Safe Answer  
 │  
 ▼  
  Gradio Chat Interface

\---

\#\#  Features

✅ Extracts and cleans pharmaceutical text from PDF    
✅ Builds FAISS vector index for lightning-fast retrieval    
✅ Ranks results contextually using a cross-encoder    
✅ Detects unsafe queries (dosage/prescription) via classifier    
✅ Works \*\*offline (retrieval)\*\* or \*\*online (RAG with OpenAI)\*\*    
✅ Gradio chatbot interface with instant feedback    
✅ Informational responses only — medically safe and ethical    
✅ Optional synonyms/fuzzy matching for misspelled queries  

\---

\#\#  Example Interactions

\*\*User:\*\* What is an excipient?    
\*\*PharmaBot:\*\*    
\> “Excipient — a substance, other than the active ingredient, that ensures safety and stability in a medicine’s formulation.”    
\> \*Note: Informational only — not medical advice.\*

\---

\*\*User:\*\* What dose of paracetamol should I take?    
\*\*PharmaBot:\*\*    
\>  I cannot provide dosage or prescription advice. Please consult a licensed pharmacist or doctor.

\---

\#\#  Workflow Summary

| Step | Description |  
|------|--------------|  
| 1️⃣ | Upload \`Pharmacy Dictionary.pdf\` |  
| 2️⃣ | Extract & clean text |  
| 3️⃣ | Split text into 513 chunks |  
| 4️⃣ | Generate embeddings (MiniLM-L6-v2) |  
| 5️⃣ | Build FAISS index |  
| 6️⃣ | Train intent classifier |  
| 7️⃣ | Launch chatbot (Gradio) |  
| 8️⃣ | Optionally enable RAG with OpenAI key |

\---

\#\#  Installation and Setup

\#\#\# 1️⃣ Clone Repository  
\`\`\`bash  
git clone https://github.com/Impawan07/Pharmacy-Chatbot-using-Retrieval-Augmented-Generation.git  
cd Pharmacy-Chatbot-using-Retrieval-Augmented-Generation

### **2️⃣ Install Dependencies**

pip install \-r requirements.txt  
\# or manually:  
pip install sentence-transformers faiss-cpu gradio openai nltk scikit-learn PyPDF2

### **3️⃣ Run Notebook**

Open `Pharmabot_Final.ipynb` in **Google Colab** or **Jupyter Notebook**, and execute cells sequentially.

### **4️⃣ Launch Chatbot**

iface.launch(share=True)

Colab will provide a **public Gradio link** where you can chat live with the bot.

---

## **📂 Repository Structure**

Pharmacy-Chatbot-using-Retrieval-Augmented-Generation/  
│  
├── Pharmabot\_Final.ipynb          \# Main Notebook  
├── cleaned\_chunks.csv             \# Preprocessed text chunks  
├── cleaned\_embs.npy               \# Sentence embeddings  
├── faiss\_cleaned.bin              \# FAISS index  
├── intent\_classifier.joblib       \# Saved classifier  
├── README.md                      \# Project documentation  
└── /assets/                       \# Optional screenshots

---

## **Disclaimer**

This project is for **educational and informational purposes only**.  
 It **does not** provide medical, dosage, or prescription advice.  
 Always consult a licensed healthcare professional before taking any medication.

---

##  **Future Enhancements**

| Feature | Description |
| ----- | ----- |
|  Multilingual Support | Extend chatbot to Hindi, Spanish, French |
|  Voice Interface | Add Speech-to-Text and Text-to-Speech |
|  Expanded Dataset | Integrate WHO \+ DrugBank \+ FDA sources |
|  Deployment | Host on Streamlit / Hugging Face Spaces |
| Query Analytics | Capture user intents for continuous improvement |

---

##  **Author**

 **M G SAI PAWAN YADAV**  
 📍 Bellary, Karnataka  
 🔗 [GitHub: Impawan07](https://github.com/Impawan07)  
 🔗 [LinkedIn: linkedin.com/in/pawanyadavsaimg](https://www.linkedin.com/in/pawanyadavsaimg)

---

##  **Acknowledgements**

* WHO Collaborating Centre for Pharmaceutical Pricing & Reimbursement Policies

* SentenceTransformers, Hugging Face, FAISS, Gradio, and OpenAI

* Project mentor team for valuable guidance

---

##  

## **License**

This project is released under the **MIT License**.  
 You may use, modify, and distribute it freely with attribution.

---

⭐ **If you find this project helpful, please star the repository\!**

https://github.com/Impawan07/Pharmacy-Chatbot-using-Retrieval-Augmented-Generation.git

Made THIS using OpenAI, Hugging Face, and FAISS.