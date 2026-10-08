# 📞 Zryth Voice Agent

A highly scalable, low-latency phone voice agent designed for Indian languages. 

Zryth Voice Agent leverages state-of-the-art AI to handle inbound calls via a standard phone number. Callers speak in their native language (e.g., English, Hindi), and the agent provides intelligent responses with **extremely low perceived latency (~700 ms - 1.2 s)**.

This system is built using **LiveKit Agents**, **Sarvam API** (for Indic STT and TTS), **Google Gemini / Groq** (for LLM reasoning), and **Supabase (PostgreSQL + pgvector)** for robust memory and RAG capabilities.

---

## ✨ Features

- **Real-Time Low Latency**: Tuned STT → LLM → TTS pipeline optimized for minimal turnaround times.
- **Multilingual Support**: Supports Indian languages via Sarvam (Saaras STT & Bulbul TTS). Handles English-Indic code-mixing natively.
- **Dynamic RAG System**: Vector-based knowledge retrieval (via pgvector and Gemini Embeddings) to provide context-aware answers to callers.
- **Lead Capture & Call State Management**: Intelligently extracts customer information (name, email, requirements) during the conversation and logs it directly to Supabase.
- **Warm Transfers**: Can transfer active calls directly to human agents via LiveKit SIP outbound trunks.
- **Complete Call Analytics**: Full transcripts and conversation logs are automatically stored for analysis.

---

## 🏗️ Architecture Stack

- **Orchestration**: LiveKit Agents Framework (WebRTC & Session States).
- **Telephony**: LiveKit SIP Trunk (Inbound origination & Outbound transfers).
- **VAD**: Silero VAD (running locally for rapid end-of-utterance detection).
- **Speech-to-Text (STT)**: Sarvam Saaras (Codemixed Indic STT).
- **LLM Engine**: Groq (Qwen) as primary for speed, with Google Gemini 2.5 Flash-Lite as fallback.
- **Vector Search & Memory**: Supabase PostgreSQL with `pgvector`.
- **Text-to-Speech (TTS)**: Sarvam Bulbul.

For detailed architecture flow, see the [Architecture Overview](architecture_overview.md).

---

## 🚀 Quickstart

### Prerequisites

You need accounts and API keys for:
1. **LiveKit** (Cloud or Self-hosted) + SIP provider (e.g., Vobiz).
2. **Sarvam AI** (for STT and TTS).
3. **Groq / Google AI** (for LLM and Embeddings).
4. **Supabase** (for Database and pgvector).

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Zryth-Voice-agent
   ```

2. **Configure Environment:**
   Copy the example environment file and fill in your actual API keys and endpoints:
   ```bash
   cp .env.example .env
   ```
   *Edit `.env` to include your LiveKit, Sarvam, Groq/Gemini, and Supabase credentials.*

3. **Install Dependencies:**
   Set up a Python virtual environment and install the required packages:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the Agent Worker:**
   Start the main agent process that will listen for LiveKit SIP jobs:
   ```bash
   python agent.py start
   ```

5. **(Optional) Run Webhook Server:**
   To process knowledge base ingestion and RAG updates:
   ```bash
   python webhook_server.py
   ```

---

## 🗂️ Project Structure

- `agent.py`: Main entrypoint for LiveKit worker and agent session loop.
- `tools.py`: Tool implementations for RAG (`search_knowledge`), Lead Capture (`capture_lead`), and Call Actions (`end_call`, `transfer_to_human`).
- `database.py`: Supabase connection and query logic for calls, messages, and RAG.
- `knowledge_ingest.py` / `webhook_server.py`: Services for document processing, embedding generation, and ingestion.
- `prompts.py`: System instructions and dynamic contexts for the LLM.
- `architecture_overview.md`: Detailed system sequence diagrams and database schemas.

---

## 🤝 Customizing the Agent

Zryth Voice Agent is highly adaptable. You can integrate your own business logic by:
1. **Adding Knowledge**: Uploading documents into the Supabase RAG system. The agent will automatically query it to answer domain-specific questions.
2. **Updating Prompts**: Modify `prompts.py` to change the agent's persona and core instructions.
3. **Expanding Tools**: Add custom Python functions inside `tools.py` and register them with the agent to hook into external APIs.

## 📄 License
Licensed under the MIT License. See [LICENSE](LICENSE) for details.
