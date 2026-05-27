# Local Agentic Data Pipeline: Autonomous Text-to-SQL Architecture with Self-Healing Guardrails

An enterprise-grade, 100% offline, secure Agentic Data Pipeline that translates natural language intents into highly accurate, safe, and executable SQL queries. By leveraging open-source LLMs running locally, this architecture eliminates data privacy compliance risks and cloud API licensing costs while ensuring deterministic execution.

## 🏗️ System Architecture

The pipeline uses a decoupled, layered approach to handle data ingestion, semantic context mapping, security vetting, and cooperative multi-agent execution:

1. **Context Retrieval Layer (RAG):** Evaluates user intent and queries a local persistent vector store to fetch relevant schema metadata.
2. **Multi-Agent Consensus Network:** A specialized Writer Agent drafts the SQL syntax, while an independent Auditor Agent performs peer code review.
3. **AST Guardrail Layer:** Intercepts the generated SQL statement and scans it using static Abstract Syntax Tree (AST) token validation before it touches data.
4. **Self-Healing Runtime Loop:** Executes the query against the database engine, intercepts live database exceptions, and streams runtime error tracebacks back into the LLM context for automated correction.

---

## 💡 Key Design Philosophies

### 1. Zero-Cloud Dependency & 100% Privacy
Built entirely on top of **Ollama** using the highly efficient **Llama 3.2** model. Zero corporate data leaves the host infrastructure, making this architecture inherently compliant with strict data privacy regulations (such as SOC2, HIPAA, or banking sector standards).

### 2. Framework Provider Decoupling
By anchoring orchestration to **LangChain**, the underlying model layer remains fully modular. Transitioning from a local open-source LLM runtime to an enterprise cloud API model (e.g., Google Gemini, Anthropic Claude, or Azure OpenAI) requires modifying exactly one line of engine initialization code, causing zero friction in downstream business logic.

### 3. Deterministic Over Creative Outputs
To handle structured operational tasks natively, the system enforces a strict `temperature=0.0` configuration combined with **Pydantic structured output validation models**, guaranteeing that the AI outputs valid JSON schemas rather than unpredictable, verbose text.

---

## 📂 Project Structure

``` text
enterprise-ai-agent/
├── chroma_db_storage/         # Local ChromaDB persistent vector storage directory (Auto-generated)
├── venv/                      # Python local virtual environment
├── .env                       # Environment configuration file (API keys if applicable)
├── setup_vector_db.py         # Connects, vectorizes, and indexes database schema metadata into ChromaDB
├── agent.py                   # Connects user query to ChromaDB and outputs a raw structured SQL layout
├── self_correcting_agent.py   # Simulates a live SQLite instance with an automated error correction loop
├── guarded_agent.py           # Secures execution via an AST token parsing safety interceptor
├── multi_agent_system.py      # Establishes a state-driven Writer-Auditor review collaboration loop
└── README.md                  # System architectural blueprint
```

## 🛡️ Security Posture & Guardrails

To prevent both **AI Prompt Injection** and malicious **SQL Injection** attacks, the pipeline deploys a strict, automated defensive barrier via `sqlparse`:

* **Mutation Blocking:** Automatically throws runtime exceptions and blocks execution if destructive mutation keywords (`DROP`, `DELETE`, `UPDATE`, `ALTER`, `TRUNCATE`) are present within the generated code block.
* **Context Boundary Enforcement:** Statically extracts statement token identifiers to verify that the query references *only* the specific database tables retrieved by the vector context layer, actively stopping model access to system configuration or parallel tables.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* [Ollama](https://ollama.com/) installed and running locally

### Installation & Setup

1. **Clone and Enter the Workspace Directory:**
```bash
cd enterprise-ai-agent

```

2. **Activate Your Virtual Environment:**
```bash
source venv/bin/activate
# On Windows use: venv\Scripts\activate

```

3. **Pull the Local LLM Core Engine:**
```bash
ollama pull llama3.2
```
4. **Seed the Local ChromaDB Vector Database:**
```bash
python setup_vector_db.py
```
5. **Test and Run the Pipeline Modules:**

* Run the basic context retrieval agent:
```bash
python agent.py
```

* Test the runtime self-healing error correction loop:
``` bash
python self_correcting_agent.py
```

* Test the static AST security token interceptor:
```bash
python guarded_agent.py
```

* Run the full multi-agent developer-auditor consensus network:
```bash
python multi_agent_system.py
```

## 🛠️ Core Technologies
* **Orchestration:** LangChain, LangChain-Ollama
* **Local LLM Engine:** Ollama (Llama 3.2)
* **Vector Database:** ChromaDB
* **Data Validation:** Pydantic v2
* **Security & AST Parsing:** SQLParse
* **Local Compute Engine:** SQLite3
