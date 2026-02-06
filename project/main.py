import os
import pandas as pd
import torch

from gemini.subject_extraction import extract_subject
from model.embeddings import compute_hospital_embed, load_or_create_embed
from model.ranking_hospital import get_query_embedding, extract_hospital_rank_k, show_hospital_k

def main():
    user_texts = input("증상을 입력해주세요 :")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    hospital_info_path = os.path.join(BASE_DIR, "data", "hospital_info.csv")
    hospital_info = pd.read_csv(hospital_info_path)
    
    embed_path = os.path.join(BASE_DIR,"cache","hospital_embeddings(sroberta-sts-normal,address0.2).pt")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("jhgan/ko-sroberta-sts")
    hospital_embed = load_or_create_embed(hospital_info, model, embed_path)
    
    query_embed = get_query_embedding(user_texts)
    candidate_hos = extract_hospital_rank_k(10,hospital_info,query_embed, hospital_embed)
    show_hospital_k(10,candidate_hos)
    
if __name__ == "__main__":
    main()