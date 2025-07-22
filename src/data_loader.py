import json
import pandas as pd
from pathlib import Path

def load_data(applicants_path, jobs_path, prospects_path):
    with open(prospects_path, 'r', encoding='utf-8') as f:
        prospects_raw = json.load(f)

    prospects_list = []
    for job_id, data in prospects_raw.items():
        for prospect in data.get("prospects", []):
            prospects_list.append({
                "job_id": int(job_id),
                "applicant_id": int(prospect["codigo"]),
                "status": prospect["situacao_candidado"],
                "comentario": prospect.get("comentario", "")
            })
    prospects_df = pd.DataFrame(prospects_list)

    applicant_ids = set(prospects_df["applicant_id"].astype(str))
    job_ids = set(prospects_df["job_id"].astype(str))

    with open(applicants_path, 'r', encoding='utf-8') as f:
        applicants_raw = json.load(f)

    applicants_filtered = []
    for aid in applicant_ids:
        if aid in applicants_raw:
            a = applicants_raw[aid]
            applicants_filtered.append({
                "applicant_id": int(aid),
                "nivel_academico": a["formacao_e_idiomas"].get("nivel_academico", ""),
                "nivel_ingles": a["formacao_e_idiomas"].get("nivel_ingles", ""),
                "nivel_espanhol": a["formacao_e_idiomas"].get("nivel_espanhol", ""),
                "cv_texto": a.get("cv_pt", ""),
                "area_atuacao": a["informacoes_profissionais"].get("area_atuacao", ""),
                "conhecimentos": a["informacoes_profissionais"].get("conhecimentos_tecnicos", "")
            })
    applicants_df = pd.DataFrame(applicants_filtered)

    with open(jobs_path, 'r', encoding='utf-8') as f:
        jobs_raw = json.load(f)

    jobs_filtered = []
    for jid in job_ids:
        if jid in jobs_raw:
            j = jobs_raw[jid]
            perfil = j["perfil_vaga"]
            jobs_filtered.append({
                "job_id": int(jid),
                "nivel_profissional": perfil.get("nivel profissional", ""),
                "nivel_ingles_job": perfil.get("nivel_ingles", ""),
                "nivel_espanhol_job": perfil.get("nivel_espanhol", ""),
                "sap": j["informacoes_basicas"].get("vaga_sap", ""),
                "atividades": perfil.get("principais_atividades", ""),
                "competencias": perfil.get("competencia_tecnicas_e_comportamentais", "")
            })
    jobs_df = pd.DataFrame(jobs_filtered)

    df = prospects_df.merge(applicants_df, on="applicant_id", how="left")
    df = df.merge(jobs_df, on="job_id", how="left")
    df.dropna(subset=["status", "cv_texto", "atividades", "competencias"], inplace=True)
    return df
