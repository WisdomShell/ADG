import json
import openai
import os
from tqdm import tqdm

# set OpenAI API 
client = openai.OpenAI(
    base_url='',
    api_key=''
)

####################################Configuration####################################################
INPUT_FILE = '/path/to/your/dataset/gpt4.json'
OUTPUT_FILE = '/path/to/your/Analyse/classifition/classified_data.jsonl'
# CHECKPOINT_FILE records the current processing progress to enable resuming from a checkpoint.
# If this is the first run, CHECKPOINT_FILE can be empty.
CHECKPOINT_FILE = '/path/to/your/checkpoint.txt'
BATCH_SIZE = 32
SAVE_INTERVAL = 1024
#####################################################################################################

CATEGORIES = [
    "Math",
    "Knowledge",
    "Reasoning",
    "Writing",
    "Coding",
    "Summarization",
    "Translation",
    "Discrete_Decision",
    "Instruction_Following",
    "Others"
]

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(index))

def classify_instruction(instruction):
    prompt = f"""Please classify the following instruction into ONE of these categories:

Categories:
- Math: Mathematical problems, calculations, equations
- Knowledge: Factual questions, general knowledge queries
- Reasoning: Logic puzzles, analytical thinking, problem-solving
- Writing: Creative writing, article writing, content generation
- Coding: Programming tasks, code generation, debugging
- Summarization: Text summarization, content condensing
- Translation: Language translation tasks
- Discrete_Decision: Categorization, labeling tasks
- Instruction_Following: Following specific instructions or procedures
- Others: Tasks that don't fit the above categories

Instruction: {instruction}

Please respond with ONLY the category name (e.g., "Math" or "Coding"). No other text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in classifying instructions. Always respond with only one category name from the given list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )
        
        category = response.choices[0].message.content.strip()
        
        
        if category not in CATEGORIES:
            category_lower = category.lower()
            for cat in CATEGORIES:
                if cat.lower() in category_lower or category_lower in cat.lower():
                    return cat
            return "Others"
        
        return category
        
    except Exception as e:
        print(f"Error during classification: {e}")
        return "Others"

def classify_batch(batch_data):
    results = []
    for entry in batch_data:
        instruction = entry['instruction']
        category = classify_instruction(instruction)
        entry['category'] = category
        results.append(entry)
    
    return results

def append_to_jsonl(data_list, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    print("Loading data...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_count = len(all_data)
    print(f"Total entries: {total_count}")
    
    start_index = load_checkpoint()
    print(f"Starting from index: {start_index}")
    
    if start_index == 0 and os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        print(f"Cleared existing output file: {OUTPUT_FILE}")
    
    
    processed_count = start_index
    buffer = []  
    
    
    with tqdm(total=total_count, initial=start_index, desc="Classifying") as pbar:
        
        for i in range(start_index, total_count, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_count)
            batch_data = all_data[i:batch_end]
            
            
            classified_batch = classify_batch(batch_data)
            buffer.extend(classified_batch)
            processed_count += len(classified_batch)
            
            
            pbar.update(len(classified_batch))
            
            
            if len(buffer) >= SAVE_INTERVAL or processed_count >= total_count:
                append_to_jsonl(buffer, OUTPUT_FILE)
                save_checkpoint(processed_count)
                print(f"\nSaved {len(buffer)} entries. Total processed: {processed_count}/{total_count}")
                buffer = []  
    
    
    if buffer:
        append_to_jsonl(buffer, OUTPUT_FILE)
        save_checkpoint(processed_count)
        print(f"\nSaved final {len(buffer)} entries.")
    
    
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    
    print("\n" + "=" * 80)
    print("CLASSIFICATION COMPLETED")
    print("=" * 80)
    print(f"Total entries processed: {processed_count}")
    print(f"Output saved to: {OUTPUT_FILE}")
    
    
    category_counts = {cat: 0 for cat in CATEGORIES}
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            category_counts[data['category']] += 1
    
    print("\nCategory Distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / processed_count) * 100
        print(f"  {cat:25s}: {count:6d} ({percentage:5.2f}%)")

if __name__ == "__main__":
    main()