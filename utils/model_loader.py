import spacy
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from spacy.matcher import PhraseMatcher
from rake_nltk import Rake
import yake
from keybert import KeyBERT
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_models():
    nlp = spacy.load('en_core_web_lg')
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    rake_extractor = Rake()
    yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=15, features=None)
    kw_model = KeyBERT()
    ner_model_name = "Nucha/Nucha_SkillNER_BERT"
    ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
    ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
    ner_pipeline_model = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
    return skill_extractor, rake_extractor, yake_extractor, kw_model, ner_pipeline_model