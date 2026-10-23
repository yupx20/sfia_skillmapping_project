import os
import sys
import logging
from flask import Flask, render_template, request, jsonify

# Ensure app directory is at the front of sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from web_scraper import scrape_job_url
    from inference_engine import SFIAInferenceEngine
except ImportError:
    from app.web_scraper import scrape_job_url
    from app.inference_engine import SFIAInferenceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    template_folder=os.path.join(BASE_DIR, "templates")
)

# Configure paths (supports both app/ internal assets and fallback to researches/)
def resolve_path(rel_path):
    p1 = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(os.path.dirname(BASE_DIR), "researches", rel_path)
    if os.path.exists(p2):
        return p2
    return p1

CS_SFIA_PATH = resolve_path(os.path.join("output", "CS", "skills_extracted_sfia_CS.xlsx"))
IS_SFIA_PATH = resolve_path(os.path.join("output", "IS", "skills_extracted_sfia_IS.xlsx"))
SFIA_META_PATH = resolve_path(os.path.join("data", "sfia9_en2025.xlsx"))

# Preload the best model inference engine
logger.info("Initializing SFIA Best Model Inference Engine...")
engine = SFIAInferenceEngine(
    cs_sfia_path=CS_SFIA_PATH,
    is_sfia_path=IS_SFIA_PATH,
    sfia_meta_path=SFIA_META_PATH
)
logger.info("Engine successfully initialized and ready for requests.")

SAMPLE_JOBS = [
    {
        "id": "fullstack_dev",
        "title": "Senior Full Stack Software Engineer",
        "company": "Tech Innovations Ltd.",
        "cluster": "CS",
        "description": (
            "We are seeking an experienced Full Stack Software Engineer to build and scale our web applications.\n"
            "Responsibilities:\n"
            "• Design, develop, test, and deploy robust web services and frontend applications.\n"
            "• Build scalable backend microservices using Python, FastAPI, Django, and Node.js.\n"
            "• Architect relational and NoSQL databases using PostgreSQL, MongoDB, and Redis.\n"
            "• Implement CI/CD pipelines with GitHub Actions, Docker, and Kubernetes on AWS.\n"
            "• Collaborate in an Agile/Scrum environment with cross-functional product and design teams.\n"
            "• Conduct code reviews, unit testing, and automated integration tests to ensure software quality and security standards.\n"
            "Requirements:\n"
            "• 4+ years of software engineering experience in JavaScript/TypeScript, React, Python, and cloud infrastructure.\n"
            "• Strong understanding of RESTful APIs, system architecture, database optimization, and Git version control."
        )
    },
    {
        "id": "data_scientist",
        "title": "Data Scientist & Machine Learning Specialist",
        "company": "Analytics Nexus Group",
        "cluster": "CS",
        "description": (
            "Looking for a passionate Data Scientist to lead our predictive modeling and AI initiatives.\n"
            "Responsibilities:\n"
            "• Develop, evaluate, and deploy machine learning and NLP models into production environments.\n"
            "• Perform exploratory data analysis, feature engineering, and statistical hypothesis testing on massive datasets.\n"
            "• Build ETL and data pipeline automation using Apache Spark, Airflow, and BigQuery.\n"
            "• Design interactive dashboards and reporting systems using Tableau and Power BI.\n"
            "• Research cutting-edge LLMs, Transformer architectures, and vector search techniques.\n"
            "Requirements:\n"
            "• Bachelor's or Master's in Computer Science, Data Science, or related quantitative field.\n"
            "• Proficiency in Python (pandas, scikit-learn, PyTorch/TensorFlow) and advanced SQL.\n"
            "• Experience in data visualization, statistical analysis, and machine learning deployment."
        )
    },
    {
        "id": "business_analyst",
        "title": "Lead IT Business Systems Analyst",
        "company": "Enterprise Solutions Corp.",
        "cluster": "IS",
        "description": (
            "Enterprise Solutions Corp is looking for a Lead IT Business Systems Analyst to bridge business requirements and technology systems.\n"
            "Responsibilities:\n"
            "• Elicit, analyze, and document complex business requirements, functional specifications, and user stories.\n"
            "• Map business processes (BPMN), identify workflow inefficiencies, and propose technological transformation solutions.\n"
            "• Manage stakeholder relationships, lead sprint planning, and facilitate user acceptance testing (UAT).\n"
            "• Collaborate with enterprise architects to evaluate ERP and CRM software vendor solutions.\n"
            "• Define project scope, track KPIs, and manage change management plans for IT system implementations.\n"
            "Requirements:\n"
            "• Degree in Information Systems, Business Administration, or Computer Science.\n"
            "• Strong analytical, communication, requirement engineering, and stakeholder management skills.\n"
            "• Experience with Jira, Confluence, Visio, and enterprise systems implementation."
        )
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sample-jobs", methods=["GET"])
def get_sample_jobs():
    return jsonify({"success": True, "samples": SAMPLE_JOBS})

@app.route("/api/scrape", methods=["POST"])
def scrape_job():
    data = request.get_json() or {}
    url = data.get("url", "").strip()
    
    if not url:
        return jsonify({"success": False, "error": "Please provide a valid Job URL."}), 400
        
    try:
        result = scrape_job_url(url)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Scraping error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Scraping failed: {str(e)}. Please copy and paste the job description text manually."
        }), 500

@app.route("/api/analyze", methods=["POST"])
def analyze_job():
    data = request.get_json() or {}
    job_title = data.get("job_title", "").strip() or "Untitled Job"
    job_description = data.get("job_description", "").strip()
    cluster = data.get("cluster", "auto").strip()
    threshold = float(data.get("threshold", 0.20))
    model_type = data.get("model_type", "skillner_qe_tfidf").strip()
    
    if not job_description:
        return jsonify({"success": False, "error": "Job description cannot be empty."}), 400
        
    try:
        analysis_result = engine.predict(
            job_title=job_title,
            job_description=job_description,
            cluster=cluster,
            threshold=threshold,
            model_type=model_type
        )
        return jsonify({"success": True, "result": analysis_result})
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Analysis failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "best_model": "SkillNER + QE + KeyBERT + TF-IDF with SFIA Skill Level Expansion",
        "cs_entries": len(engine.sfia_cs_df),
        "is_entries": len(engine.sfia_is_df)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" SFIA Skill Mapping Web Application Server")
    print(f" Best Model: SkillNER + QE + KeyBERT + TF-IDF + Expansion")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
