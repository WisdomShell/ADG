import torch
import re
import numpy as np
import json
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple
from tqdm import tqdm
import pickle

# Configuration parameters
model_name = "/path/to/your/qwen2.5-7b"
INPUT_JSONL = "/path/to/your/qwen2.5_generation_alpaca_gpt4.jsonl"
OUTPUT_DIR = "/path/to/your/qwen_alpaca_gpt4" 
EMBEDDINGS_PATH = "/path/to/your/embedding_qwen_alpaca_gpt4.pkl"
CLUSTERS_PATH = "/path/to/your/cluster_qwen_alpaca_gpt4.pkl"
CHECKPOINT_DIR = "/path/to/your/checkpoints_Qwen_alpaca_gpt4"

# Metric weight configuration
METRIC_1 = "Sc.cov_trace" 
WEIGHT_1 = 0.6
METRIC_2 = "Sc.SED" 
WEIGHT_2 = 0.4
FINAL_SELECT_COUNT = 10000
SAVE_EVERY_N_BATCHES = 512

# ==================== GPU Setup ====================
def setup_gpu():
    """Setup single GPU"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")
    return device

# ==================== Model Loading ====================
def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer - single GPU version"""
    print(f"Loading model {model_name}...")
    
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
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
    model = model.to(device)
    
    return tok, model

# ==================== Utility Functions ====================
def content_weights(input_ids, tok, a=1e-3, bonus_entity=1.3):
    """Simple SIF with number/unit/entity boosting"""
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
def answer_vec_new(instruction: str, answer: str, tok, model, device, layers=(-4, -3, -2)):
    """
    Generate answer vector by concatenating instruction and answer,
    then pooling over the entire token sequence
    """
    if not answer.strip():
        hidden_size = getattr(model.config, 'hidden_size', 3840)
        return np.zeros(hidden_size, dtype=np.float32)
    
    try:
        inst_toks = tok(instruction, add_special_tokens=False, return_tensors="pt")
        ans_toks = tok(answer, add_special_tokens=False, return_tensors="pt")
        
        inst_len = inst_toks["input_ids"].shape[1]
        ans_len = ans_toks["input_ids"].shape[1]
        
        full_text = instruction + " " + answer
        full_toks = tok(
            full_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
            padding=False
        )
        
        full_toks = {k: v.to(device) for k, v in full_toks.items()}
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            out = model(**full_toks)
        
        hs = out.hidden_states
        L = len(hs)
        sel = [L + i for i in layers]
        
        H = torch.stack([hs[i][0] for i in sel], dim=0).mean(0)
        
        total_len = full_toks["input_ids"].shape[1]
        answer_start_idx = inst_len
        
        H_answer = H

        if H_answer.shape[0] == 0:
            hidden_size = H.shape[1]
            return np.zeros(hidden_size, dtype=np.float32)
        
        answer_input_ids = full_toks["input_ids"][0].cpu().tolist()
        w_np = content_weights(answer_input_ids, tok)
        w = torch.tensor(w_np, device=device, dtype=torch.float16).unsqueeze(1)
        
        Hw = (H_answer * w).sum(dim=0) / torch.clamp(w.sum(), min=1e-8)
        v = Hw / torch.clamp(Hw.norm(p=2), min=1e-12)
        
        return v.detach().cpu().numpy().astype(np.float32)
        
    except Exception as e:
        print(f"Error generating answer vector: {e}")
        hidden_size = getattr(model.config, 'hidden_size', 3840)
        return np.zeros(hidden_size, dtype=np.float32)

# ==================== Data Loading ====================
def load_all_data(input_jsonl):
    """Load all data from JSONL file"""
    print(f"Reading file {input_jsonl}...")
    
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error at line {line_num}: {e}")
    
    print(f"Total {len(data)} items loaded")
    
    return data

def load_embeddings_and_clusters(embeddings_path, clusters_path):
    """Load existing embeddings and clustering results"""
    print(f"Loading embeddings from {embeddings_path}...")
    
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Successfully loaded {len(embeddings)} embeddings")
    print(f"Loading clustering results from {clusters_path}...")
    
    with open(clusters_path, 'rb') as f:
        cluster_labels = pickle.load(f)
    
    print(f"Successfully loaded clustering results")
    
    return embeddings, cluster_labels

# ==================== Gram Matrix and Metrics ====================
def build_S(v_list):
    """Build Gram matrix"""
    V = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in v_list], axis=0)
    return V @ V.T

def _sym(M): 
    return 0.5*(M+M.T)

def _eigvals_psd(M):
    w = np.linalg.eigvalsh(_sym(M))
    return np.clip(np.sort(w)[::-1], 0.0, None)

def _center_gram(S):
    K = S.shape[0]
    H = np.eye(K) - np.ones((K,K))/K
    return H @ _sym(S) @ H

def _mean_cos(S):
    """Calculate mean cosine similarity"""
    K = S.shape[0]
    if K <= 1:
        return 0.0
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    return float(S[mask].mean())

def selected_metrics_from_S(S: np.ndarray, eps: float = 1e-12) -> Dict[str, float]:
    """Calculate 7 selected diversity metrics"""
    S = _sym(np.asarray(S, float))
    K = S.shape[0]
    assert K >= 2 and S.shape[1] == K

    lam_S = _eigvals_psd(S)
    sec_S = float(lam_S[0] / (lam_S.sum() + eps))
    mean_cos = _mean_cos(S)

    Sc = _center_gram(S)
    lam_Sc = _eigvals_psd(Sc)
    tr_Sc = lam_Sc.sum()
    if tr_Sc < eps:
        sed_Sc = 0.0; hspec_Sc = 0.0; cov_trace = 0.0; r12_Sc = float("inf")
    else:
        sed_Sc = 1.0 - float(lam_Sc[0] / tr_Sc)
        p = lam_Sc / tr_Sc
        hspec_Sc = float(-np.sum(p*np.log(p+eps)) / np.log(K))
        cov_trace = float(tr_Sc / K)
        r12_Sc = float(lam_Sc[0] / (lam_Sc[1] + eps))

    def cov_trace_of_index_subset(idx):
        S_sub = S[np.ix_(idx, idx)]
        Sc_sub = _center_gram(S_sub)
        lam = _eigvals_psd(Sc_sub)
        return float(lam.sum() / max(len(idx),1))
    
    base = cov_trace
    lood_var = 0.0
    if K > 2 and base > eps:
        vals = []
        for k in range(K):
            idx = [i for i in range(K) if i!=k]
            vals.append(cov_trace_of_index_subset(idx))
        lood_var = max(abs(base - v) for v in vals) / (base + eps)

    return {
        "Sc.SED": sed_Sc,
        "Sc.H_spec": hspec_Sc,
        "S.SEC": sec_S,
        "S.mean_cos": mean_cos,
        "Sc.cov_trace": cov_trace,
        "Sc.eig_ratio_12": r12_Sc,
        "LOOD_var": lood_var,
    }

# ==================== Data Processing ====================
def process_single_item(item, tok, model, device):
    """Process single JSONL item - generate vectors for 5 samples and compute metrics"""
    instruction = item.get("instruction", "")
    generation_answers = item.get("generated_answers", [])
    
    if len(generation_answers) != 5:
        if len(generation_answers) < 5:
            generation_answers.extend([""] * (5 - len(generation_answers)))
        else:
            generation_answers = generation_answers[:5]
    
    answer_vecs = []
    for answer in generation_answers:
        vec = answer_vec_new(instruction, str(answer), tok, model, device, layers=(-4, -3, -2))
        answer_vecs.append(vec)
    
    S = build_S(answer_vecs)
    metrics = selected_metrics_from_S(S)
    
    output_item = item.copy()
    output_item.update(metrics)
    
    return output_item

# ==================== Checkpoint Management ====================
def save_checkpoint(processed_data, batch_num):
    """Save checkpoint"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_batch_{batch_num}.json")
    
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_file}, {len(processed_data)} items")

def load_latest_checkpoint():
    """Load latest checkpoint"""
    if not os.path.exists(CHECKPOINT_DIR):
        return None, 0
    
    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint_batch_")])
    
    if not checkpoint_files:
        return None, 0
    
    latest_file = checkpoint_files[-1]
    batch_num = int(latest_file.split("_")[-1].replace(".json", ""))
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_file)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    print(f"Loaded {len(processed_data)} processed items, resuming from batch {batch_num}")
    
    return processed_data, batch_num

# ==================== Cluster-based Selection ====================
def calculate_combined_score(item, metric_1=METRIC_1, weight_1=WEIGHT_1, metric_2=METRIC_2, weight_2=WEIGHT_2):
    """Calculate combined score"""
    value_1 = item.get(metric_1, 0)
    value_2 = item.get(metric_2, 0)
    
    combined_score = weight_1 * value_1 + weight_2 * value_2
    return combined_score

def cluster_based_selection(processed_data, cluster_labels, final_count=FINAL_SELECT_COUNT):
    """Perform proportional selection based on clustering"""
    print(f"Starting cluster-based proportional selection, target: {final_count} items...")
    
    cluster_groups = {}
    for i, (data_item, cluster_id) in enumerate(zip(processed_data, cluster_labels)):
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(data_item)
    
    print(f"Total {len(cluster_groups)} clusters")
    
    for item in processed_data:
        item["combined_score"] = calculate_combined_score(item)
    
    cluster_sizes = {cid: len(items) for cid, items in cluster_groups.items()}
    total_items = sum(cluster_sizes.values())
    
    selection_ratio = final_count / total_items
    print(f"Total {total_items} items, target {final_count}, ratio: {selection_ratio:.3f} ({selection_ratio*100:.1f}%)")
    
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
    
    print(f"Cluster results: Top {len(cluster_top_data)}, Middle {len(cluster_middle_data)}, Bottom {len(cluster_bottom_data)}")
    
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
        
        print(f"Saved {len(group_data)} items from {group_name} group to {output_file}")
        
        if group_data:
            metric_1_values = [item.get(METRIC_1, 0) for item in group_data]
            metric_2_values = [item.get(METRIC_2, 0) for item in group_data]
            combined_scores = [item["combined_score"] for item in group_data]
            
            print(f"  {METRIC_1} range: {min(metric_1_values):.4f} - {max(metric_1_values):.4f}")
            print(f"  {METRIC_2} range: {min(metric_2_values):.4f} - {max(metric_2_values):.4f}")
            print(f"  Combined score range: {min(combined_scores):.4f} - {max(combined_scores):.4f}")

# ==================== Main Function ====================
def main():
    device = setup_gpu()
    
    print(f"Using single GPU")
    print(f"Metric weights: {METRIC_1}={WEIGHT_1}, {METRIC_2}={WEIGHT_2}")
    
    tok, model = load_model_and_tokenizer(model_name, device)
    all_data = load_all_data(INPUT_JSONL)
    embeddings, cluster_labels = load_embeddings_and_clusters(EMBEDDINGS_PATH, CLUSTERS_PATH)
    
    processed_data, start_batch = load_latest_checkpoint()
    if processed_data is None:
        processed_data = []
        start_batch = 0
    
    print(f"Starting processing, total {len(all_data)} items, from batch {start_batch}")
    
    start_idx = len(processed_data)
    remaining_data = all_data[start_idx:]
    
    print(f"Total {len(remaining_data)} items to process")
    
    batch_size = 4
    batch_count = 0
    
    for i in tqdm(range(0, len(remaining_data), batch_size), desc="Processing"):
        batch_data = remaining_data[i:i+batch_size]
    
        for item in batch_data:
            try:
                processed_item = process_single_item(item, tok, model, device)
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing data: {e}")
                continue
        
        batch_count += 1
        
        if batch_count % SAVE_EVERY_N_BATCHES == 0:
            save_checkpoint(processed_data, start_batch + batch_count)
        
        if batch_count % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Successfully processed {len(processed_data)} items")
    save_checkpoint(processed_data, start_batch + batch_count)
    cluster_based_selection(processed_data, cluster_labels)
    
    print("All processing complete!")

if __name__ == "__main__":
    main()