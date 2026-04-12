import os
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import argparse
import orjson
from pathlib import Path


MODEL_NAME = "/path/to/your/llama3-8b/"
OUTPUT_DIR = "/path/to/your/dir/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "llama_gpt4.jsonl")
NUM_RETURN_SEQUENCES = 5
BATCH_WRITE_SIZE = 1024
MAX_NEW_TOKENS = 256
TEMPERATURE = 1.4
TOP_P = 0.9



def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(gpu)
        return rank, world_size, gpu
    else:
        return 0, 1, 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_and_tokenizer(gpu):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        device_map=None  
    ).to(gpu)
    model.eval()
    return model, tokenizer


class InstructionDataset(Dataset):
    def __init__(self, data, completed_ids):
        self.data = [item for item in data if item['id'] not in completed_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'id': item['id'],
            'instruction': item['instruction'],
            'output': item.get('output', '')
        }


def generate_answers_batch(model, tokenizer, instructions, device):
    prompts = [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instr}\n\n### Response:" 
        for instr in instructions
    ]
    
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=80
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    batch_answers = []
    batch_size = len(prompts)
    
    for i in range(batch_size):
        prompt_len = inputs.input_ids[i].shape[0]
        start_idx = i * NUM_RETURN_SEQUENCES
        end_idx = (i + 1) * NUM_RETURN_SEQUENCES
        sample_outputs = outputs[start_idx:end_idx]
        
        answers = []
        for out in sample_outputs:
            decoded = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            answers.append(decoded.strip())
        
        batch_answers.append(answers)
    
    return batch_answers


def get_completed_ids_auto(output_file):
    completed_ids = set()
    output_dir = os.path.dirname(output_file)
    output_basename = os.path.basename(output_file)
    
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        completed_ids.add(item['id'])
        except Exception as e:
            print(f"Error reading main output file {output_file}: {e}")
    
    
    import glob
    temp_pattern = os.path.join(output_dir, f"{output_basename}.rank_*.tmp")
    temp_files = glob.glob(temp_pattern)
    
    for temp_file in temp_files:
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        completed_ids.add(item['id'])
        except Exception as e:
            print(f"Error reading temp file {temp_file}: {e}")
    
    return completed_ids
# -----------------------------
# main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='JSON file path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    
    rank, world_size, gpu = setup_distributed()
    device = torch.device(f'cuda:{gpu}')
    
    
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if dist.is_initialized():
        dist.barrier()  

    
    if rank == 0:
        print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(gpu)

    
    
    if rank == 0:
        print("Loading input data...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    
    if rank == 0:
        completed_ids = get_completed_ids_auto(OUTPUT_FILE)
        
        print(f"Found {len(completed_ids)} completed items")
        completed_list = list(completed_ids)
    else:
        completed_list = []
    
    
    if dist.is_initialized():
        
        if rank == 0:
            
            
            serialized_data = pickle.dumps(completed_list)
            data_size = torch.tensor(len(serialized_data), dtype=torch.long).to(device)
        else:
            data_size = torch.tensor(0, dtype=torch.long).to(device)
        
        
        dist.broadcast(data_size, src=0)
        
        if rank == 0:
            data_tensor = torch.frombuffer(serialized_data, dtype=torch.uint8).to(device)
        else:
            data_tensor = torch.zeros(data_size.item(), dtype=torch.uint8).to(device)
        
        
        dist.broadcast(data_tensor, src=0)
        
        if rank != 0:
            serialized_data = data_tensor.cpu().numpy().tobytes()
            completed_list = pickle.loads(serialized_data)
        
        completed_ids = set(completed_list)
    else:
        completed_ids = set(completed_list) if rank == 0 else set()

    
    dataset = InstructionDataset(full_data, completed_ids)
    
    if len(dataset) == 0:
        print(f"Rank {rank}: No data to process")
        cleanup_distributed()
        return
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False  
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        num_workers=0  
    )

    
    temp_output_file = f"{OUTPUT_FILE}.rank_{rank}.tmp"
    buffer = []

    
    if rank == 0:
        print(f"Starting generation with {len(dataset)} items across {world_size} GPUs")
    
    for batch in tqdm(dataloader, desc=f"Rank {rank}", disable=(rank != 0)):
        try:
            ids = batch['id']
            instructions = batch['instruction']
            reference_outputs = batch['output']
            
            
            if torch.is_tensor(ids[0]):
                ids = [id.item() for id in ids]
            else:
                ids = list(ids)

            answers_list = generate_answers_batch(model, tokenizer, instructions, device)

            for i in range(len(ids)):
                buffer.append({
                    "id": ids[i],
                    "instruction": instructions[i],
                    "output": reference_outputs[i],
                    "generated_answers": answers_list[i]
                })

            
            if len(buffer) >= BATCH_WRITE_SIZE:
                with open(temp_output_file, "a", encoding="utf-8") as f:
                    for entry in buffer:
                        f.write(orjson.dumps(entry).decode('utf-8') + "\n")
                buffer.clear()
                
        except Exception as e:
            print(f"Error in rank {rank}: {e}")
            continue

    
    if buffer:
        with open(temp_output_file, "a", encoding="utf-8") as f:
            for entry in buffer:
                f.write(orjson.dumps(entry).decode('utf-8') + "\n")

    
    if dist.is_initialized():
        dist.barrier()

    
    if rank == 0:
        print("Merging temporary files...")
        with open(OUTPUT_FILE, "a", encoding="utf-8") as final_f:
            for r in range(world_size):
                temp_file = f"{OUTPUT_FILE}.rank_{r}.tmp"
                if os.path.exists(temp_file):
                    with open(temp_file, "r", encoding="utf-8") as temp_f:
                        final_f.write(temp_f.read())
                    os.remove(temp_file)  
        print(f"Generation completed. Results saved to {OUTPUT_FILE}")

    cleanup_distributed()
    print(f"Rank {rank} finished.")

# -----------------------------
if __name__ == "__main__":
    main()
