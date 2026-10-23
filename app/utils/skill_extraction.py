from utils.model_loader import load_models
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
skill_extractor, rake_extractor, yake_extractor, kw_model = load_models()

def extract_skills_skillner(text):
    if not isinstance(text, str) or not text.strip():
            return []
            
    try:
        annotations = skill_extractor.annotate(text)
        skills = set()
        
        results = annotations.get('results', {})
        if not results:
            return []

        for match in results.get('full_matches', []):
            skills.add(match['doc_node_value'].lower())
            
        for match in results.get('ngram_scored', []):
            skills.add(match['doc_node_value'].lower())
            
        return list(skills)
    
    except Exception as e:
        print(f"Terjadi error saat memproses teks: '{str(text)[:50]}...'. Error: {e}")
        return []

# Technical proper nouns, tools, and terms that should NEVER be expanded into English slang/homonyms
TECH_NO_EXPAND = {
    'git', 'docker', 'bug', 'pipeline', 'shell', 'spring', 'spark', 'ant', 'rest', 
    'react', 'rust', 'go', 'node', 'cookies', 'server', 'patch', 'terminal', 'driver',
    'python', 'java', 'c', 'c++', 'sql', 'nosql', 'r', 'matlab', 'ruby', 'perl', 'php',
    'aws', 'azure', 'gcp', 'jira', 'redis', 'kafka', 'hadoop', 'linux', 'windows',
    'devops', 'kubernetes', 'cloud', 'ci/cd', 'agile', 'scrum', 'kanban'
}

# Slang, pejoratives, offensive words, and inappropriate homonyms from WordNet to filter out
PROFANITY_SLANG_BLACKLIST = {
    'rotter', 'dirty dog', 'rat', 'skunk', 'stinker', 'stinkpot', 'bum', 'puke', 
    'crumb', 'lowlife', 'scum bag', 'so and so', 'bastard', 'bitch', 'grapevine', 
    'stevedore', 'dockworker', 'dockhand', 'lumper', 'dock worker', 'bugger', 
    'jackass', 'fool', 'churl', 'ass', 'bitchy', 'jerk', 'sod'
}

def expand_terms(term):
    term_lower = term.lower().strip()
    # Skip expansion for technical tools, programming languages, and short abbreviations
    if term_lower in TECH_NO_EXPAND or len(term_lower) <= 3:
        return set()

    synonyms = set()
    for syn in wn.synsets(term_lower):
        syn_def = syn.definition().lower()
        if any(bad in syn_def for bad in ['offensive', 'vulgar', 'contemptible', 'slang', 'insult', 'despicable', 'insect']):
            continue
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ').lower()
            if name != term_lower and name not in PROFANITY_SLANG_BLACKLIST and len(name) > 2:
                synonyms.add(name)
    return synonyms

def extract_skills_skillner_qe(text):

    original_skills = extract_skills_skillner(text)
    expanded_skills = set(original_skills)

    for skill in original_skills:
        synonyms = expand_terms(skill)
        expanded_skills.update(synonyms)

    return list(expanded_skills)

def extract_rake_keywords(text):
    if not isinstance(text, str) or not text.strip():
        return []
    rake_extractor.extract_keywords_from_text(text)
    keywords = rake_extractor.get_ranked_phrases()
    return [kw.strip().lower() for kw in keywords]

def extract_yake_keywords(text):
    if not isinstance(text, str): return []
    keywords = yake_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def extract_keybert_keywords(text):
    if not isinstance(text, str) or not text.strip(): return []
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', use_mmr=True)
    clean_kws = []
    for kw, score in keywords:
        kw_lower = kw.lower()
        if not any(bad in kw_lower for bad in PROFANITY_SLANG_BLACKLIST):
            clean_kws.append(kw)
    return clean_kws