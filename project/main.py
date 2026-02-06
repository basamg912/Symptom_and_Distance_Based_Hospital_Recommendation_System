import os
import pandas as pd
import torch
import sys

# Ensure project root is in sys.path if running strictly as script
# We need to add the parent of 'project' to sys.path so that 'from project.x import y' works.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.gemini.subject_extraction import extract_subject
from project.model.embeddings import compute_hospital_embed, load_or_create_embed
from project.model.ranking_hospital import get_query_embedding, extract_hospital_rank_k, show_hospital_k
from project.utils import config
from project.model.loader import get_model

def main():
    user_texts = input("증상을 입력해주세요 :")
    
    hospital_info_path = config.DATA_DIR / "hospital_info.csv"
    hospital_info = pd.read_csv(hospital_info_path)
    
    embed_path = config.CACHE_DIR / "hospital_embeddings(sroberta-sts-normal,address0.2).pt"

    model = get_model("jhgan/ko-sroberta-sts")
    hospital_embed = load_or_create_embed(hospital_info, model, embed_path)
    
    query_embed = get_query_embedding(user_texts)
    candidate_hos = extract_hospital_rank_k(10,hospital_info,query_embed, hospital_embed)
    show_hospital_k(10,candidate_hos)
    
if __name__ == "__main__":
    main()