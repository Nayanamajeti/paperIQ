# 🔬 PaperIQ — Research Paper Intelligence Assistant

> Upload any academic paper. Get instant structured analysis, cited answers, and cross-paper comparison — powered by RAG + Gemini AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-1.5--flash-orange?style=flat-square&logo=google)
![FAISS](https://img.shields.io/badge/FAISS-Vector--Store-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🧠 What Is PaperIQ?

Most RAG apps let you "chat with a PDF." PaperIQ goes further.

It automatically extracts **what actually matters** from any research paper — novelty, methodology, limitations, real-world impact — and lets you ask deep questions with **exact page-level citations**. It also self-corrects when it can't find an answer, and supports **cross-paper comparison** across multiple uploaded documents.

Built as a domain-agnostic research tool — works equally well for medical, engineering, finance, law, or any academic field.

---

## ✨ Features

### 📋 Auto Paper Brief
Upload a paper and instantly get a structured 7-point analysis:
- **Research Problem** — what gap does this paper address?
- **Novelty** — what's new that didn't exist before?
- **Methodology** — models, datasets, techniques used
- **Key Findings** — results with exact numbers
- **Limitations** — weaknesses the authors acknowledge
- **Future Work** — what the authors suggest next
- **Real World Impact** — how this could be applied in practice

### 💬 Cited Q&A
Ask any question about your paper and get:
- A precise, grounded answer (no hallucinations)
- Source citations with **exact file name + page number**
- Expandable snippet previews of the retrieved context

### 🔄 Self-Correcting Answers
If the system can't find an answer, it automatically rewrites your question using query rewriting and retries — before giving up. Inspired by Corrective RAG architecture.

### ⚖️ Cross-Paper Comparison
Upload multiple papers and compare them:
- "Which paper's methodology is more robust?"
- "How do these papers differ in their approach to X?"
- "Which study achieved better accuracy and why?"

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              Streamlit UI                   │
│   (Ask Questions / Paper Brief / Compare)   │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│            RAG Pipeline                     │
│                                             │
│  Query → HuggingFace Embeddings             │
│        → FAISS Vector Search (top-k chunks) │
│        → Gemini 1.5 Flash (answer gen)      │
│        → Self-Correction (if needed)        │
│        → Source Citation Extraction         │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           Document Ingestion                │
│                                             │
│  PDF → PyPDF Loader → Text Chunking         │
│      → HuggingFace Embeddings               │
│      → FAISS Index (saved locally)          │
└─────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A free Google Gemini API key → [Get one here](https://aistudio.google.com)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/paperiq.git
cd paperiq

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> ⚠️ Never commit your `.env` file. It's already in `.gitignore`.

### Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
paperiq/
│
├── app.py                  # Streamlit UI — tabs, upload, display
├── rag_pipeline.py         # Core RAG logic, self-correction, citations
├── ingest.py               # PDF loading, chunking, FAISS indexing
│
├── data/                   # Drop your PDFs here (gitignored)
├── faiss_index/            # Auto-generated vector index (gitignored)
│
├── .env                    # Your API key (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| RAG Framework | LangChain + LangChain-Community |
| PDF Processing | PyPDF |
| UI | Streamlit |
| Environment | python-dotenv |

---

## 🔬 Novel Contributions

Unlike standard PDF chatbots, PaperIQ introduces:

1. **Structured Paper Intelligence** — automatic 7-point academic analysis, not just Q&A
2. **Self-Correcting RAG** — query rewriting on failed retrievals, inspired by Corrective RAG research
3. **Cross-Document Reasoning** — explicit multi-paper comparison with per-paper attribution
4. **Domain Agnostic** — no domain-specific fine-tuning; works across all academic fields

---

## 📊 Sample Output

**Question:** *What is novel about this paper?*

**Answer:**
> The paper introduces a hybrid approach combining FFDNet for Gaussian noise removal with Richardson-Lucy deconvolution for motion blur, unified under a VGG16-based domain classifier. Prior work addressed each degradation type in isolation — this is the first pipeline to automatically detect and route images to the appropriate restoration model without manual intervention. Evaluated on PSNR and SSIM metrics across 4 degradation categories.

**Sources:**
> 📄 paper.pdf | Page 3 — *"Unlike existing methods that require manual selection of the restoration technique..."*
> 📄 paper.pdf | Page 7 — *"The domain classifier achieves 94.2% routing accuracy across noise, blur, color distortion..."*

---

## 🛣️ Roadmap

- [ ] ArXiv paper search integration (search without uploading)
- [ ] Export Paper Brief as PDF report
- [ ] Support for `.docx` and `.txt` files
- [ ] Chat history with memory across questions
- [ ] LangGraph multi-agent upgrade for deeper reasoning

---

## 👩‍💻 Author

**Nayana Majeti**
B.Tech — AI & ML | Vishwakarma Institute of Technology, Pune
AWS Certified AI Practitioner | Azure AI Fundamentals
📧 nayana.majeti@gmail.com | [LinkedIn](https://linkedin.com/in/nayana-majeti)

---

## 📄 License

MIT License — feel free to use, modify, and build on top of this.

---

> *Built as part of a personal initiative to explore applied LLM engineering in academic research workflows.*
