# Zryth Voice Agent System Architecture

## 1. Database Architecture (Supabase PostgreSQL)

The system relies on a Supabase PostgreSQL backend, utilizing standard relational tables for logging and `pgvector` for semantic search.

```mermaid
erDiagram
    CALLS ||--o{ MESSAGES : "has many"
    CALLS {
        uuid id PK "Primary Key"
        string livekit_room "LiveKit Room ID"
        string phone "Caller's Phone Number (Caller ID)"
        string customer_name "Captured via Lead Tool"
        string email "Captured via Lead Tool"
        string company "Captured via Lead Tool"
        string requirement "Captured via Lead Tool"
        string language "Conversation Language Code"
        timestamp started_at "Call Start Time"
        timestamp ended_at "Call End Time"
        integer duration_seconds "Total Call Length"
    }

    MESSAGES {
        uuid id PK "Primary Key"
        uuid call_id FK "References CALLS"
        string speaker "customer or maya"
        text message "Transcribed text or generated response"
        timestamp created_at "Message Timestamp"
    }

    KNOWLEDGE {
        uuid id PK "Primary Key"
        text content "Document Chunk Text"
        vector embedding "Google Gemini Embedding Vector"
        jsonb metadata "Source Document Metadata"
    }
```

### Database Features
- **Call State Management:** A unique row is inserted into `CALLS` upon connection. Tools incrementally `UPDATE` this row as they capture lead requirements.
- **Transcripts:** Every transcription chunk and LLM output is asynchronously appended to the `MESSAGES` table for full historical recall on the dashboard.
- **Knowledge Base (RAG):** The `KNOWLEDGE` table utilizes the `pgvector` extension. A custom PostgreSQL RPC function (`match_knowledge`) performs cosine similarity comparisons against the caller's embedded search query to retrieve the closest text chunks.

---

## 2. Tools & Integrations Architecture

The system uses a best-in-class cascaded pipeline designed for extremely low latency (Time-to-First-Byte) and human-like interactions in Indian contexts.

### Component Integrations
- **Core Orchestration:** **LiveKit Agents Framework** runs the core async Python loop, managing WebRTC streams and session states.
- **Telephony (SIP):** **LiveKit SIP Trunk** handles routing standard inbound phone calls into WebRTC rooms, and dialing out to human agents (`transfer_to_human` tool).
- **Voice Activity Detection (VAD):** **Silero VAD** (running locally). This is tuned heavily to filter background noise while still capturing snappy human interruptions.
- **Speech-to-Text (STT):** **Sarvam Saaras STT**. Chosen for its native understanding of English-Indic code-mixing (speaking English words amidst Hindi/Telugu sentences).
- **LLM (The Brain):** **Google Gemini 2.5 Flash-Lite**. Chosen for its incredibly fast inference speeds (TTFT) which is vital for voice agents. It routes prompts, processes context, and handles tool invocations.
- **Embeddings:** **Google Gemini (gemini-embedding-2)**. Generates semantic vectors for incoming RAG queries.
- **Text-to-Speech (TTS):** **Sarvam Bulbul TTS**. Generates ultra-realistic Indian-accented speech that streams directly back into the LiveKit pipeline.

---

## 3. System Flow Diagram

The following diagram maps the lifecycle of a phone call from the moment the user dials in, to how Maya processes audio and executes backend tools.

```mermaid
sequenceDiagram
    autonumber
    actor Caller
    participant Telecom as Telecom Network
    participant LiveKit as LiveKit Cloud (SIP Trunk)
    participant Agent as EC2 (Maya Python Agent)
    participant AI as Gemini & Sarvam APIs
    participant DB as Supabase (Postgres)

    %% Connection Flow
    Caller->>Telecom: Dials Zryth Phone Number
    Telecom->>LiveKit: Routes call via SIP
    LiveKit->>Agent: Triggers `entrypoint`
    
    %% Setup
    Agent->>DB: `create_call()` - Logs Caller ID
    Agent->>LiveKit: Mounts `GreeterAgent` (English fallback)
    Agent-->>Caller: Speaks Welcome Greeting
    
    %% Audio Loop
    rect rgb(240, 240, 240)
        Note right of Caller: Continuous Audio Streaming Loop
        Caller->>Agent: Speaks (Audio Stream)
        Agent->>Agent: Silero VAD detects end of utterance
        Agent->>AI: Sarvam STT transcribes audio
        AI-->>Agent: Returns text (e.g. "What is your software?")
        Agent->>DB: `save_message()` saves transcript
        
        %% LLM Thinking & Tool Execution
        Agent->>AI: Gemini 2.5 processes transcript
        
        alt RAG Tool Triggered
            AI-->>Agent: Tool Call `search_knowledge`
            Agent->>AI: Gemni embeds query
            Agent->>DB: `match_knowledge` (Vector Similarity)
            DB-->>Agent: Returns top 5 Text Chunks
            Agent->>AI: Gemini formulates answer based on DB chunks
        end
        
        alt Lead Tool Triggered
            AI-->>Agent: Tool Call `capture_lead`
            Agent->>DB: `update_call_lead()` (Upserts CRM info)
        end
        
        %% Speech Output
        AI-->>Agent: Returns Generated Text
        Agent->>DB: `save_message()` saves Maya's response
        Agent->>AI: Sarvam TTS converts text to speech chunks
        AI-->>Agent: Returns Audio Stream
        Agent-->>Caller: Plays Audio
    end
    
    %% Hangup Flow
    Caller->>Agent: "Goodbye"
    Agent-->>Caller: Speaks Goodbye Message
    Agent->>Agent: Tool Call `end_call()`
    Agent->>DB: `finish_call()` (Logs duration)
    Agent->>LiveKit: Force job shutdown
    LiveKit-->>Telecom: Ends SIP Session
```
