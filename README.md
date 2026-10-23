# SFIA Skill Mapping & Competency Intelligence

An end-to-end Natural Language Processing (NLP) framework and interactive web application for extracting IT technical skills from real-world job descriptions and automatically mapping them into the standardized **SFIA 9 (Skills Framework for the Information Age)** competency taxonomy across Levels 1 to 7.

---

## Table of Contents
- [1. Research Overview](#1-research-overview)
  - [Problem Statement](#problem-statement)
  - [Dataset](#dataset)
  - [5-Stage NLP Methodology](#5-stage-nlp-methodology)
- [2. Research Results & Benchmarks](#2-research-results--benchmarks)
  - [Computer Science (CS) Cluster](#computer-science-cs-cluster)
  - [Information Systems (IS) Cluster](#information-systems-is-cluster)
  - [Key Research Takeaways](#key-research-takeaways)
- [3. The Web Application](#3-the-web-application)
  - [Core Features](#core-features)
  - [Supported Model Pipelines](#supported-model-pipelines)
  - [User Interface Highlights](#user-interface-highlights)
- [4. Project Structure](#4-project-structure)
- [5. Installation & Usage](#5-installation--usage)
  - [Prerequisites](#prerequisites)
  - [Option A: Local Execution (Python / Conda)](#option-a-local-execution-python--conda)
  - [Option B: Containerized Execution (Docker & Docker Compose)](#option-b-containerized-execution-docker--docker-compose)
- [6. API Reference](#6-api-reference)
- [7. Authors & Acknowledgments](#7-authors--acknowledgments)

---

## 1. Research Overview

### Problem Statement
In the rapidly evolving software and IT landscape, job postings on platforms like LinkedIn and JobStreet describe candidate requirements using informal keywords, company-specific jargon, and emerging tooling acronyms. Conversely, global IT competence frameworks—most notably **SFIA 9**—standardize competencies into formal, descriptive responsibility statements stratified across 7 responsibility levels.

This research establishes an automated semantic bridging system to:
1. **Extract** technical and soft IT skills from unstructured, multi-lingual job postings.
2. **Expand** domain terminology to resolve acronyms and technical synonyms.
3. **Map** extracted competencies to discrete SFIA skill-level definitions using semantic vector spaces.
4. **Hierarchically infer** foundational proficiency levels based on SFIA's cumulative competency model.

---

### Dataset
The research was conducted across **1,391 real-world job postings** collected via web scraping:

| Dataset Segment | Count | Source Portals | Description |
| :--- | :--- | :--- | :--- |
| **Computer Science (CS)** | 767 | LinkedIn, JobStreet | Software Engineers, DevOps, AI/ML Engineers, Full-Stack Developers, Cloud Architects |
| **Information Systems (IS)** | 624 | LinkedIn, JobStreet | Business Analysts, ERP/CRM Consultants, IT Project Managers, Systems Analysts |
| **SFIA 9 Framework** | 147 Skills | SFIA Foundation 2025 | 6 Categories, 672 discrete level statements (Levels 1–7) |
| **Ground Truth (GT)** | Multi-Annotator | Domain Experts | Validated via Cohen's / Fleiss Kappa inter-annotator agreement |

---

### 5-Stage NLP Methodology

```
[Raw Job Postings / URLs]
           │
           ▼
Stage 1: Text Cleaning & Normalization
  ├── HTML stripping, email/phone redaction, unicode cleanup
  └── Lowercasing and whitespace normalization
           │
           ▼
Stage 2: Multilingual Preprocessing & Translation
  ├── Indonesian language detection & translation (googletrans)
  └── Stopwords removal and lemmatization (NLTK)
           │
           ▼
Stage 3: Multi-Model Skill Extraction
  ├── Baseline: SkillNER (Exact dictionary & PhraseMatcher)
  ├── Acronym & Synonym Enrichment: WordNet Query Expansion (QE)
  └── Statistical/Semantic Filtering: RAKE, YAKE, and KeyBERT (all-MiniLM-L6-v2)
           │
           ▼
Stage 4: Vectorization & Cosine Similarity Mapping
  ├── Unified TF-IDF vector space across job skills & 672 SFIA level descriptions
  ├── Cosine similarity scoring with adaptive thresholding (default: 0.20)
  └── SFIA Hierarchical Skill Level Expansion (Level N ──► Levels 1 ... N-1)
           │
           ▼
Stage 5: Evaluation against Domain Expert Ground Truth
  └── Precision, Recall, and F1-Score benchmarking
```

---

## 2. Research Results & Benchmarks

The research evaluated 8 distinct model combinations with and without **SFIA Hierarchical Level Expansion**.

### Computer Science (CS) Cluster

| Model Pipeline | Expanded | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SkillNER QE_KeyBERT** | **True** | **413** | **190** | **22** | **68.49%** | **94.94%** | **79.58% (Best F1)** |
| SkillNER QE | True | 434 | 232 | 1 | 65.17% | 99.77% | 78.84% |
| SkillNER_KeyBERT | True | 413 | 206 | 22 | 66.72% | 94.94% | 78.37% |
| SkillNER | True | 395 | 189 | 40 | 67.64% | 90.80% | 77.53% |
| SkillNER YAKE_KeyBERT | True | 338 | 123 | 97 | 73.32% | 77.70% | 75.45% |
| SkillNER YAKE | True | 333 | 120 | 102 | 73.51% | 76.55% | 75.00% |
| SkillNER QE | False | 423 | 230 | 12 | 64.78% | 97.24% | 77.76% |
| SkillNER QE_KeyBERT | False | 344 | 163 | 91 | 67.85% | 79.08% | 73.04% |
| SkillNER | False | 342 | 182 | 93 | 65.27% | 78.62% | 71.32% |
| SkillNER RAKE | True | 312 | 132 | 123 | 70.27% | 71.72% | 70.99% |
| SkillNER RAKE_KeyBERT | True | 269 | 84 | 166 | 76.20% | 61.84% | 68.27% |

---

### Information Systems (IS) Cluster

| Model Pipeline | Expanded | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SkillNER QE_KeyBERT** | **True** | **412** | **192** | **18** | **68.21%** | **95.81%** | **79.69% (Best F1)** |
| SkillNER_KeyBERT | True | 418 | 215 | 12 | 66.03% | 97.21% | 78.65% |
| SkillNER QE | True | 428 | 237 | 2 | 64.36% | 99.53% | 78.17% |
| SkillNER | True | 384 | 197 | 46 | 66.09% | 89.30% | 75.96% |
| SkillNER QE | False | 413 | 235 | 17 | 63.73% | 96.05% | 76.62% |
| SkillNER YAKE | True | 311 | 124 | 119 | 71.49% | 72.33% | 71.91% |
| SkillNER | False | 338 | 181 | 92 | 65.13% | 78.60% | 71.23% |
| SkillNER YAKE_KeyBERT | True | 302 | 127 | 128 | 70.40% | 70.23% | 70.31% |
| SkillNER QE_KeyBERT | False | 312 | 160 | 118 | 66.10% | 72.56% | 69.18% |
| SkillNER RAKE | True | 293 | 127 | 137 | 69.76% | 68.14% | 68.94% |
| SkillNER RAKE_KeyBERT | True | 246 | 93 | 184 | 72.57% | 57.21% | 63.98% |

---

### Key Research Takeaways

1. **Overall Champion (`SkillNER + QE + KeyBERT + Expansion`):**
   * Produces the highest overall F1-score across both domains (**79.58% in CS** and **79.69% in IS**).
   * KeyBERT's Maximal Marginal Relevance (MMR) ranking effectively eliminates noisy terms while retaining semantically dense keywords.
2. **Near-Perfect Sensitivity (`SkillNER + QE`):**
   * Achieved **99.77% recall in CS** and **99.53% recall in IS** (only 1 to 2 missed ground truth competencies out of hundreds).
   * Serves as the recommended default for applications where false negatives must be minimized.
3. **The Power of SFIA Hierarchical Level Expansion:**
   * In SFIA theory, achieving Level $N$ implies baseline mastery of foundational levels $1 \dots N-1$.
   * Explicitly modeling this domain rule improved F1-scores across all pipelines by **+6.2% to +8.5%**, substantially reducing False Negatives.

---

## 3. The Web Application

To translate research into practice, the repository includes an interactive single-page web application (`app/`) that performs live scraping and instantaneous SFIA competency mapping.

### Core Features

* **Live Web Scraping:** Paste any public job link from **LinkedIn** or **JobStreet**; the backend automatically extracts the job title, company name, and full description text.
* **Direct Text Input & Sample Jobs:** Paste job descriptions manually or load pre-curated test roles (Senior Full Stack Engineer, Data Scientist, Lead Business Systems Analyst).
* **Interactive Model Switcher:** Switch between the top-performing research models directly in the UI.
* **Sensitivity Threshold Slider:** Dynamically adjust the cosine similarity threshold (0.05 to 0.50) to evaluate match strictness in real time.
* **JSON Export:** Download structured mapping outputs including matched tokens, similarity scores, category allocations, and level hierarchies.

---

### Supported Model Pipelines

| Pipeline Option | Composition | Best Used For |
| :--- | :--- | :--- |
| **`SkillNER + QE + TF-IDF`** *(Default)* | SkillNER + WordNet QE + TF-IDF | Cleanest vocabulary, near-perfect recall (99%+), and fastest inference. |
| **`SkillNER + QE + KeyBERT + TF-IDF`** | SkillNER + WordNet QE + KeyBERT + TF-IDF | Benchmark champion maximizing F1-score (79.6%). |
| **`SkillNER + TF-IDF`** | SkillNER Exact Taxonomy + TF-IDF | Baseline matching with zero synonym expansion. |

---

### User Interface Highlights

* **SFIA 6-Category Radar Chart:** Visualizes competency balance across the 6 SFIA 9 core domains (*Strategy and architecture, Change and transformation, Development and implementation, Delivery and operation, People and skills, Relationships and engagement*).
* **Proficiency Distribution Chart:** Breaks down required competency seniority from Level 1 (*Follow*) to Level 7 (*Set strategy, inspire, mobilize*).
* **Hierarchical Level Nodes:** Visual badges (`L1` ... `L7`) highlighting both the directly targeted proficiency level and all hierarchically expanded prerequisite levels.

---

## 4. Project Structure

```
sfiaskillmapping_project/
├── app/                              # Production Web Application
│   ├── app.py                        # Flask API Server & route controller
│   ├── inference_engine.py           # Preloaded SFIA Inference Engine
│   ├── web_scraper.py                # LinkedIn & JobStreet web scraper
│   ├── data/
│   │   └── sfia9_en2025.xlsx         # Master SFIA 9 competency definitions
│   ├── output/
│   │   ├── CS/skills_extracted_sfia_CS.xlsx  # 672 CS extracted level entries
│   │   └── IS/skills_extracted_sfia_IS.xlsx  # 672 IS extracted level entries
│   ├── utils/                        # NLP preprocessing, SkillNER & model loaders
│   ├── static/                       # Custom CSS styling & vanilla JS frontend
│   ├── templates/index.html          # Single-page UI template
│   ├── Dockerfile                    # Optimized Linux container definition
│   ├── docker-compose.yml            # App-level container orchestrator
│   └── requirements.txt              # Frozen Python dependencies
│
├── researches/                       # Academic Research & Experiments
│   ├── data/                         # Raw job datasets (CSJobs, ISJobs) & annotations
│   ├── output/                       # Evaluation benchmarks, visual charts & radars
│   ├── scraping/                     # Experimental scraping notebooks & cookies
│   ├── step_1_cleaning.py            # Dataset cleaning script
│   ├── step_2_preprocessing.py       # Translation & preprocessing script
│   ├── step_3_extraction.py          # 8-model skill extraction experiments
│   ├── step_4_mapping.py             # Cosine similarity & level expansion mapping
│   ├── step_5_evaluation.py          # Ground-truth metric evaluation
│   └── yake_param_test.py            # YAKE hyperparameter tuning script
│
├── docker-compose.yml                # Root Docker Compose (points to ./app)
├── .dockerignore                     # Docker build exclusions
├── .gitignore                        # Git repository exclusions
├── LICENSE                           # MIT License
└── README.md                         # Documentation & Research Report
```

---

## 5. Installation & Usage

### Prerequisites
* **Python:** 3.11 recommended (avoid Python 3.14+ due to compiled wheel compatibility).
* **Miniconda / Anaconda** (recommended on Windows).
* **Docker & Docker Compose** (optional, for containerized execution).

---

### Option A: Local Execution (Python / Conda)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yupx20/sfia_skillmapping_project.git
   cd sfia_skillmapping_project
   ```

2. **Activate your Python 3.11 environment:**
   ```powershell
   # If using Miniconda:
   conda activate <your-env-name>
   ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r app/requirements.txt
   ```

4. **Launch the application:**
   From the repository root:
   ```powershell
   python app/app.py
   ```
   Or from inside the `app/` folder:
   ```powershell
   cd app
   python app.py
   ```

5. **Access the application:**
   Open your browser at `http://127.0.0.1:5000`.

---

### Option B: Containerized Execution (Docker & Docker Compose)

The application includes an optimized Docker container that pre-downloads NLTK corpora, installs CPU-only PyTorch (saving >3.5 GB of CUDA bloat), and serves the app via **Gunicorn WSGI**:

1. **Build and start the container:**
   ```bash
   docker compose up --build
   ```

2. **Access the application:**
   Open `http://localhost:5000`.

3. **Stop the container:**
   ```bash
   docker compose down
   ```

---

## 6. API Reference

### Analyze Job Description
* **Endpoint:** `POST /api/analyze`
* **Content-Type:** `application/json`

**Request Body:**
```json
{
  "job_title": "Full Stack Engineer",
  "job_description": "Proficient in React, Node.js, Python, PostgreSQL, and AWS cloud infrastructure.",
  "cluster": "CS",
  "threshold": 0.15,
  "model_type": "skillner_qe_tfidf"
}
```

**Response Structure (Summary):**
```json
{
  "success": true,
  "result": {
    "job_title": "Full Stack Engineer",
    "detected_cluster": "CS",
    "active_model": "SkillNER + QE + TF-IDF",
    "summary": {
      "total_extracted_skills": 12,
      "total_sfia_competencies_matched": 4,
      "highest_similarity_score": 0.285
    },
    "radar_data": {
      "labels": ["Strategy and architecture", "Development and implementation", ...],
      "counts": [0, 3, 1, 0, 0, 0]
    },
    "level_distribution": {
      "Level 1": 4, "Level 2": 4, "Level 3": 3, "Level 4": 2, "Level 5": 1
    },
    "mapped_skills": [
      {
        "skill": "Programming/software development",
        "category": "Development and implementation",
        "highest_level": 5,
        "highest_score": 0.285,
        "expanded_levels": [1, 2, 3, 4, 5],
        "matched_tokens": ["react", "node.js", "python", "postgresql"],
        "descriptions": {
          "5": "Sets standards for programming tools and techniques..."
        }
      }
    ]
  }
}
```

### Scrape Job Posting URL
* **Endpoint:** `POST /api/scrape`
* **Content-Type:** `application/json`

**Request Body:**
```json
{
  "url": "https://www.linkedin.com/jobs/view/..."
}
```

---

## 7. Authors & Acknowledgments

* **Lead Researcher:** Rizal / yupx20
* **Framework:** SFIA 9 by the [SFIA Foundation](https://sfia-online.org/)
* **Python:** 3.11 (Flask)
* **NLP Libraries:** spaCy, SkillNER, KeyBERT, HuggingFace Sentence-Transformers, NLTK, Scikit-learn