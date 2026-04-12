import torch
import re
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


#################################Configuration############################################
model_name = "/path/to/your/llama3-8b"
INPUT_JSONL = "/path/to/your/llama3_generation_alpaca_gpt4.jsonl"
OUTPUT_DIR = "/path/to/your/filtered_alpaca_gpt4" 
EMBEDDINGS_PATH = "/path/to/your/embedding_alpaca_gpt4.pkl"
CLUSTERS_PATH = "/path/to/your/cluster_alpaca_gpt4.pkl"
K_CLUSTERS = 1000
DEVICE = "cuda:0"

WEIGHT_COV_TRACE = 0.6
WEIGHT_SED = 0.4
FINAL_SELECT_COUNT = 10000
##############################################################################################


def load_model_and_tokenizer(model_name, device):
    
    print(f"load {model_name} to {device}...")
    
    torch.set_num_threads(1)
    
    tok = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True,
        local_files_only=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
    model = model.to(device)
    return tok, model

def content_weights(input_ids, tok, a=1e-3, bonus_entity=1.3):
    w = np.ones(len(input_ids), dtype=np.float32)
    for i, tid in enumerate(input_ids):
        try:
            tok_str = tok.decode([int(tid)]).strip()
            if re.search(r"[\d%℃°]", tok_str):
                w[i] *= bonus_entity
            if len(tok_str) == 0 or re.fullmatch(r"[\s,.;:!?()\[\]{}\"'`~\-_/\\]+", tok_str):
                w[i] = 0.0
        except:
            w[i] = 1.0
    return w

@torch.no_grad()
def sentence_vec(text: str, tok, model, device, layers=(-4, -3, -2)):
    if not text.strip():
        return np.zeros(model.config.hidden_size, dtype=np.float32)
    
    try:
        with torch.cuda.device(device):
            toks = tok(
                text, 
                return_tensors="pt", 
                add_special_tokens=False, 
                truncation=True, 
                max_length=1024,
                padding=False
            )
            
            toks = {k: v.to(device) for k, v in toks.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                out = model(**toks)
            
            hs = out.hidden_states
            L = len(hs)
            sel = [L + i for i in layers]
            H = torch.stack([hs[i][0] for i in sel], dim=0).mean(0)

            input_ids = toks["input_ids"][0].cpu().tolist()
            w_np = content_weights(input_ids, tok)
            w = torch.tensor(w_np, device=device, dtype=torch.float16).unsqueeze(1)

            Hw = (H * w).sum(dim=0) / torch.clamp(w.sum(), min=1e-8)
            v = Hw / torch.clamp(Hw.norm(p=2), min=1e-12)
            
            return v.detach().cpu().numpy().astype(np.float32)
            
    except Exception as e:
        print(f"error: {e}")
        return np.zeros(model.config.hidden_size, dtype=np.float32)

def generate_instruction_embeddings(data, tok, model, device):
    
    embeddings = []
    batch_size = 32
    
    for i in tqdm(range(0, len(data), batch_size), desc="Generating embeddings"):
        batch_data = data[i:i+batch_size]
        batch_embeddings = []
        
        for item in batch_data:
            instruction = item.get("instruction", "").strip()
            if not instruction:
                embedding = np.zeros(model.config.hidden_size, dtype=np.float32)
            else:
                embedding = sentence_vec(instruction, tok, model, device, layers=(-4, -3, -2))
            batch_embeddings.append(embedding)
        
        embeddings.extend(batch_embeddings)
        
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return embeddings

def load_all_data_and_embeddings(input_jsonl, embeddings_path, tok, model, device):
    
    print(f"read {input_jsonl}...")
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        # data = json.load(f)
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON error line {line_num}: {e}")
    
    
    if os.path.exists(embeddings_path):
        print(f"embedding: {embeddings_path}，loading...")
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"load {len(embeddings)} embeddings")
        
        if len(embeddings) != len(data):
            print(f"embedding numbers ({len(embeddings)})not same with ({len(data)}),re-generation...")
            embeddings = None
    else:
        
        embeddings = None
    
    
    if embeddings is None:
        embeddings = generate_instruction_embeddings(data, tok, model, device)
        
        
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    return data, embeddings

def perform_kmeans_clustering(embeddings, clusters_path):
    if os.path.exists(clusters_path):
        print(f"find clusters:{clusters_path}，loading...")
        with open(clusters_path, 'rb') as f:
            cluster_labels = pickle.load(f)
        print(f"finish load {K_CLUSTERS} clusters")
    else:
        print(f"K-means，K={K_CLUSTERS}...")
        embeddings_array = np.stack(embeddings, axis=0)
        
        kmeans = MiniBatchKMeans(
            n_clusters=K_CLUSTERS,
            random_state=42,
            batch_size=min(1000, len(embeddings) // 10),
            n_init=3,
            max_iter=100,
            verbose=0
        )
        
        print("clustering...")
        cluster_labels = kmeans.fit_predict(embeddings_array)
        
       
        os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
        with open(clusters_path, 'wb') as f:
            pickle.dump(cluster_labels, f)
        print(f"save clusters {clusters_path}")
    
    return cluster_labels

def build_S(v_list):
    
    V = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in v_list], axis=0)
    return V @ V.T

def S_from_texts(texts5, tok, model, device):
    
    vecs = []
    for t in texts5:
        vecs.append(sentence_vec(t, tok, model, device, layers=(-4, -3, -2)))
    S = build_S(vecs)
    return S

def _sym(M): 
    return 0.5*(M+M.T)

def _eigvals_psd(M):
    w = np.linalg.eigvalsh(_sym(M))
    return np.clip(np.sort(w)[::-1], 0.0, None)

def _center_gram(S):
    K = S.shape[0]
    H = np.eye(K) - np.ones((K,K))/K
    return H @ _sym(S) @ H

def calculate_metrics(S: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    S = _sym(np.asarray(S, float))
    K = S.shape[0]
    
    
    Sc = _center_gram(S)
    lam_Sc = _eigvals_psd(Sc)
    tr_Sc = lam_Sc.sum()
    
    if tr_Sc < eps:
        sed_Sc = 0.0
        cov_trace = 0.0
    else:
        sed_Sc = 1.0 - float(lam_Sc[0] / tr_Sc)
        cov_trace = float(tr_Sc / K)
    
    return {
        "Sc.SED": sed_Sc,
        "Sc.cov_trace": cov_trace,
    }

def process_single_item(item, tok, model, device,answers_num = 5):
    
    instruction = item.get("instruction", "")
    generation_answers = item.get("generated_answers", [])
    
    
    if len(generation_answers) != answers_num:
        if len(generation_answers) < answers_num:
            generation_answers.extend([""] * (answers_num - len(generation_answers)))
        else:
            generation_answers = generation_answers[:answers_num]
    
    
    full_texts = []
    for answer in generation_answers:
        full_text = instruction + " " + str(answer)
        full_texts.append(full_text)
    
    
    S = S_from_texts(full_texts, tok, model, device)
    
    
    metrics = calculate_metrics(S)
    
    
    output_item = item.copy()
    output_item.update(metrics)
    
    return output_item

def calculate_combined_score(item, weight_cov_trace=WEIGHT_COV_TRACE, weight_sed=WEIGHT_SED):
    cov_trace = item.get("Sc.cov_trace", 0)
    sed = item.get("Sc.SED", 0)
    
    combined_score = weight_cov_trace * cov_trace + weight_sed * sed
    return combined_score

def cluster_based_selection(processed_data, cluster_labels, final_count=FINAL_SELECT_COUNT):
    cluster_groups = {}
    for i, (data_item, cluster_id) in enumerate(zip(processed_data, cluster_labels)):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(data_item)
    
    print(f"{len(cluster_groups)} clusters")
    
    
    for item in processed_data:
        item["combined_score"] = calculate_combined_score(item)
    
    
    cluster_sizes = {cid: len(items) for cid, items in cluster_groups.items()}
    total_items = sum(cluster_sizes.values())
    
    
    selection_ratio = final_count / total_items
    print(f"all:{total_items}，filter:{final_count}，percent: {selection_ratio:.3f} ({selection_ratio*100:.1f}%)")
    
    
    cluster_top_data = []
    cluster_middle_data = []  
    cluster_bottom_data = []
    
    for cluster_id, cluster_data in cluster_groups.items():
        cluster_size = len(cluster_data)
        
        if cluster_size < 3:
            cluster_middle_data.extend(cluster_data)
            continue
        
        
        sorted_cluster_data = sorted(cluster_data, key=lambda x: x["combined_score"], reverse=True)
        
        
        cluster_top_count = max(1, int(cluster_size * selection_ratio))
        cluster_bottom_count = max(1, int(cluster_size * selection_ratio))
        cluster_middle_count = max(1, int(cluster_size * selection_ratio))
        
        
        total_selected = cluster_top_count + cluster_middle_count + cluster_bottom_count
        if total_selected > cluster_size:
            scale_factor = cluster_size / total_selected
            cluster_top_count = max(1, int(cluster_top_count * scale_factor))
            cluster_bottom_count = max(1, int(cluster_bottom_count * scale_factor))
            cluster_middle_count = cluster_size - cluster_top_count - cluster_bottom_count
            cluster_middle_count = max(0, cluster_middle_count)
        
        
        cluster_top = sorted_cluster_data[:cluster_top_count]
        cluster_bottom = sorted_cluster_data[-cluster_bottom_count:] if cluster_bottom_count > 0 else []
        
        
        if cluster_middle_count > 0:
            remaining_data = sorted_cluster_data[cluster_top_count:-cluster_bottom_count] if cluster_bottom_count > 0 else sorted_cluster_data[cluster_top_count:]
            if len(remaining_data) >= cluster_middle_count:
                start_idx = (len(remaining_data) - cluster_middle_count) // 2
                cluster_middle = remaining_data[start_idx:start_idx + cluster_middle_count]
            else:
                cluster_middle = remaining_data
        else:
            cluster_middle = []
        
        cluster_top_data.extend(cluster_top)
        cluster_middle_data.extend(cluster_middle)
        cluster_bottom_data.extend(cluster_bottom)
    
    print(f"Top:{len(cluster_top_data)}, Middle:{len(cluster_middle_data)}, Bottom:{len(cluster_bottom_data)}")
    
    
    def select_final_data(group_data, group_name):
        if len(group_data) <= final_count:
            return group_data
        
        
        if group_name == "bottom":
            sorted_group_data = sorted(group_data, key=lambda x: x["combined_score"], reverse=False)
        else:
            sorted_group_data = sorted(group_data, key=lambda x: x["combined_score"], reverse=True)
        
        return sorted_group_data[:final_count]
    
    
    final_top_data = select_final_data(cluster_top_data, "top")
    final_middle_data = select_final_data(cluster_middle_data, "middle")  
    final_bottom_data = select_final_data(cluster_bottom_data, "bottom")
    
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    
    groups = [("top", final_top_data), ("middle", final_middle_data), ("bottom", final_bottom_data)]
    for group_name, group_data in groups:
        output_file = os.path.join(OUTPUT_DIR, f"{group_name}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(group_data, f, ensure_ascii=False, indent=2)
        
        print(f"save{group_name} {len(group_data)}data to  {output_file}")
        
        
        if group_data:
            cov_trace_values = [item.get("Sc.cov_trace", 0) for item in group_data]
            sed_values = [item.get("Sc.SED", 0) for item in group_data]
            combined_scores = [item["combined_score"] for item in group_data]
            
            print(f"  Sc.cov_trace score: {min(cov_trace_values):.4f} - {max(cov_trace_values):.4f}")
            print(f"  Sc.SED score: {min(sed_values):.4f} - {max(sed_values):.4f}")
            print(f"  final score: {min(combined_scores):.4f} - {max(combined_scores):.4f}")

def main():
    
    device = DEVICE
    if not torch.cuda.is_available():
        print(" GPU error, use CPU")
        device = "cpu"
    else:
        print(f"device: {device}")
        torch.cuda.set_per_process_memory_fraction(0.9, device=int(device.split(':')[1]))
    
    print(f"Sc.cov_trace={WEIGHT_COV_TRACE}, Sc.SED={WEIGHT_SED}")
    
    
    tok, model = load_model_and_tokenizer(model_name, device)
    
    #embeddings
    all_data, embeddings = load_all_data_and_embeddings(
        INPUT_JSONL, EMBEDDINGS_PATH, tok, model, device)
    
    # K-means
    cluster_labels = perform_kmeans_clustering(embeddings, CLUSTERS_PATH)
    
    print(f" {len(all_data)} ")
    
    
    processed_data = []
    batch_size = 256
    
    for i in tqdm(range(0, len(all_data), batch_size), desc="Processing data"):
        batch_data = all_data[i:i+batch_size]
        
        for item in batch_data:
            try:
                processed_item = process_single_item(item, tok, model, device)
                processed_data.append(processed_item)
            except Exception as e:
                print(f"error: {e}")
                continue
        
        
        if i % (batch_size * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"finish {len(processed_data)} ")
    
    
    cluster_based_selection(processed_data, cluster_labels)
    
    print("finish！")

if __name__ == "__main__":
    main()