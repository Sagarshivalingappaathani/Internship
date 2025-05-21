# Week -1: Finding the Best Agentic Framework for Our Project

## What Our Project Needs (Similarity Between Procedural Documents)

- **Python and GPU support**  
  The code must run efficiently in Python and utilize GPU acceleration for fast embedding generation, graph operations, and deep learning models.

- **End-to-end orchestration**  
  A complete pipeline is required—from document ingestion and parsing, to graph construction, model training, and similarity prediction.

- **Detailed monitoring**  
  We need visibility into each step's runtime, token usage, and error tracking for easy debugging and optimization.

- **Error handling and fallback logic**  
  The system should gracefully handle exceptions and reroute low-confidence results to alternative paths or human review.

- **Large plugin ecosystem**  
  Built-in loaders, vector databases, and evaluation tools are essential to reduce boilerplate and focus on core research tasks.

---

## 🔧 Framework Options

### 1. LangChain + LangGraph + LangSmith

**Strengths:**
- 200+ ready-made connectors for file formats, databases, evaluation tools.
- LangSmith auto-records each run: shows timings, token use, and error details.
- LangGraph allows complex agent workflows as graphs—makes long or branching jobs reproducible.

**Weaknesses:**
- Heavy initial setup due to many packages and services.
- Can feel too complex for quick, simple experiments.

---

### 2. AutoGen

**Strengths:**
- Agents talk via plain language—easy to prototype and debug.
- Predefined agent roles (Planner, Coder, Critic) reduce orchestration effort.
- Supports async calls and logs OpenTelemetry traces for long-running jobs.

**Weaknesses:**
- Smaller plugin ecosystem; custom adapters may be needed.
- Manual tool registration introduces boilerplate.

---

### 3. CrewAI

**Strengths:**
- Lightweight and fast to set up—even on constrained hardware.
- No dependency on LangChain—keeps codebase lean.
- The "Crew" model fits well with phases like parsing, training, scoring.

**Weaknesses:**
- Few built-in integrations; we must implement connectors ourselves.
- Lacks built-in monitoring and error handling.
- Not ideal for branching workflows.

---

### 4. LlamaIndex

**Strengths:**
- `LlamaParse` extracts structured content from PDFs/Markdown with ease.
- Ships with many datastore connectors for quick embedding push/pull.
- Optional Llama Agents can invoke LangChain tools when needed.

**Weaknesses:**
- Focused on document ingestion + retrieval-augmented generation (RAG).
- Needs pairing with another framework for full orchestration.
- Smaller community and fewer third-party tools than LangChain.

---

## Two Approaches

### **Approach A – Mix and Match**

We combine frameworks for specific strengths:
- Use **LlamaIndex** for parsing,
- **LangChain + LangGraph** for workflow orchestration,
- **AutoGen** for experimental loops,
- **CrewAI** for smoke tests and light pipelines.

**Pros:**
- Maximum flexibility.
- Swap any component independently.
- Best tool for each job.

**Cons:**
- Must manage multiple repos and versions.
- Slightly more complex deployment and maintenance.

---

### **Approach B – Single Framework**

Choose one stack—likely LangChain—for all tasks.

**Pros:**
- Consistent pipeline from start to finish.
- Easier scaling and onboarding.
- No inter-framework compatibility issues.

**Cons:**
- Limited to available features in chosen stack.
- Vendor lock-in risk.

---

## Conclusion

We should **start with the LangChain stack** for its:
- End-to-end support,
- Built-in monitoring,
- Large ecosystem.

Later, we can:
- Use **AutoGen** or **CrewAI** for quick or lightweight experiments,
- Use **LlamaIndex** for complex document ingestion.

> This hybrid strategy balances research flexibility with production stability.
