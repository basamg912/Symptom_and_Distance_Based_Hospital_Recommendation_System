from project.model.loader import get_model
from project.gemini.subject_extraction import extract_subject
from sentence_transformers import util
import torch
import umap
import pandas as pd

def get_query_embedding(text: str):
    model = get_model()
    extract_dept = extract_subject(text)
    extract_dept= extract_dept + "성북구"
    query_embed = model.encode(extract_dept,convert_to_tensor=True)
    query_embed = query_embed.cpu()
    return query_embed

def extract_hospital_rank_k(k, hospital_info, query_embed,hospital_embed):
    hospital_rank = pd.DataFrame(columns = hospital_info.columns)
    umap_model = umap.UMAP(n_components=100)
    compare_with_query = hospital_embed.squeeze().cpu()
    compare_with_query = torch.tensor(compare_with_query).float()
    query_embed = query_embed.unsqueeze(0)
    query_embed = torch.tensor(query_embed).float()
    sim = util.cos_sim(query_embed,compare_with_query)
    
    list_of_hospital = sim.topk(k)[1].squeeze().tolist()
    
    hospital_rank = pd.DataFrame(columns = hospital_info.columns)
    for idx in list_of_hospital:
        new_row = hospital_info.loc[idx]
        hospital_rank = pd.concat([hospital_rank, new_row.to_frame().T], ignore_index=True)
    return hospital_rank

def show_hospital_k(k,hospital_rank):
    for i in range(k):
        print("="*80)
        print(f"{'-병원명-':<30}|{'-진료과-':<30}")
        print(f"{hospital_rank.loc[i,'hospital_name']:<30}{hospital_rank.loc[i,'medical_subject']:<30}")
        print(f"{'-주소-':<50}")
        print(f"{hospital_rank.loc[i,'address']:<50}")