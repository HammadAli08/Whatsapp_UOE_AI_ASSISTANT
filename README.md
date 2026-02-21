# 🎓 UOE AI Assistant – WhatsApp RAG Agent

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)
![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis)

A fully autonomous, production-grade **WhatsApp Assistant** powered by Advanced RAG (Retrieval-Augmented Generation) to answer academic queries for the **University of Education**. 

This application listens to messages via the **Meta Cloud API**, enriches user queries with short-term conversation memory, retrieves highly contextual academic policies from Pinecone vector databases, and delivers accurate answers directly into the user's WhatsApp chat.

---

## ✨ Key Features

- **📱 Native WhatsApp Integration:** Operates entirely within WhatsApp via the official Meta Cloud Webhooks API. No frontend app needed.
- **🧠 Advanced RAG Architecture:** Utilizes OpenAI `gpt-4o` for generation, and HuggingFace for dense vector embeddings combined with sparse BM25 retrieval.
- **🔄 Smart Conversation Memory:** Powered by Redis Cloud, the agent remembers the context of the conversation for up to 24 hours per user.
- **🗂️ Dynamic Namespace Switching:** Seamlessly toggle between different University knowledge bases. Text `/ms-phd` or `/rules` in WhatsApp to switch domains instantly!
- **⚡ Asynchronous Processing:** Uses FastAPI `BackgroundTasks` to guarantee sub-second webhook handshakes with Meta, preventing timeout disconnects.

---

## 🏗️ Architecture

1. **User Details:** The user texts the WhatsApp Business number.
2. **Webhook (FastAPI):** Meta sends a POST request to `/api/whatsapp/webhook`.
3. **Memory (Redis):** The agent fetches the user's previous context using their Phone Number as the `session_id`.
4. **Agentic Router:** The system identifies if the user requested a specific namespace (e.g., `bs-adp`, `ms-phd`, `rules`).
5. **Retrieval (Pinecone):** The user's query is enhanced and embedded to retrieve the top matching institutional documents.
6. **Generation (LLM):** The LLM synthesizes an accurate answer.
7. **Delivery:** The backend makes a secure API call to Meta to send the final answer back to the user's phone.

---

## 🚀 Quick Start (Local Development)

### 1. Requirements
* Python 3.12+ 
* Poetry or `uv` for dependency management
* Meta Developer Account (WhatsApp Cloud API setup)

### 2. Installation
```bash
git clone https://github.com/HammadAli08/Whatsapp_UOE_AI_ASSISTANT.git
cd Whatsapp_UOE_AI_ASSISTANT/backend

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` and fill in your secrets.
```bash
cp .env.example .env
```
Key configurations include your `OPENAI_API_KEY`, `PINECONE_API_KEY`, `REDIS_URL`, and the Meta `WHATSAPP_TOKEN`.

### 4. Run the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

---

## 🌍 Deployment on Render

This project is configured for seamless deployment on Render.com.

1. **Root Directory:** `backend`
2. **Build Command:** `./build.sh`
3. **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:** Map all variables from your `.env` directly into the Render Dashboard.

Once deployed, grab your Render URL (e.g., `https://my-app.onrender.com`) and paste it into the **Callback URL** field in your Meta Developer Console!

---

## 🎮 How to Use in WhatsApp

Once your app is Live, anyone can text your WhatsApp number! 

**Example Prompts:**
- *"What are the admission requirements for BS Computer Science?"*
- *"When does the Fall semester begin?"*

**Pro-Tip: Switching Namespaces**
You can force the AI to search a specific domain by putting a slash command at the start of your message:
- `/rules How many days of absence are allowed?`
- `/ms-phd Are there any GAT requirements for admission?`

The AI will save this preference to Redis memory and continue answering from that specific domain until you switch back!

---
*Built with ❤️ for the University of Education.*
