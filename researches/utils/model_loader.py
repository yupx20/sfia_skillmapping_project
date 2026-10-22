import spacy
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
from spacy.matcher import PhraseMatcher
from rake_nltk import Rake
import yake
from keybert import KeyBERT

def load_models():
    nlp = spacy.load('en_core_web_lg')
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    rake_extractor = Rake()
    yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.2, dedupFunc='seqm', features=None)
    kw_model = KeyBERT()
    return skill_extractor, rake_extractor, yake_extractor, kw_model