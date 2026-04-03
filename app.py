import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# --- CONFIGURATION ---
# Using ShweYon-V3-Base - Base model for Myanmar chat
MODEL_NAME = "URajinda/ShweYon-V3-Base"

# Load Dataset from GitHub
DATASET_REPO = "amkyawdev/myanmar-llm-dataset"
DATASET_PATH = "data/processed"

# System Prompt - Instructions for the AI assistant
SYSTEM_PROMPT = """သင်သည် မြန်မာစကားပြောတဲ့ AI စာရေးပါ။ မြန်မာလို ပြောပါ။
- ပါဝင်ပါတရား ဖြစ်ပါ။
- ရှင်းလင်းပါ။
- မြန်မာဘာသာစကားနဲ့ ပြောပါ။
- အမှားမလုပ်ပါ။
- သုံးစွဲသူကို ကူညီပါ။"""

# Load dataset from GitHub
print("📂 Loading Dataset from GitHub...")
try:
    dataset = load_dataset("json", data_files={
        "train": f"https://raw.githubusercontent.com/{DATASET_REPO}/main/{DATASET_PATH}/train.jsonl",
        "validation": f"https://raw.githubusercontent.com/{DATASET_REPO}/main/{DATASET_PATH}/validation.jsonl"
    })
    print(f"✅ Dataset loaded! Train: {len(dataset['train'])} samples")
except Exception as e:
    print(f"⚠️ Dataset load error: {e}")
    dataset = None

# --- MODEL LOADING ---
print("🚀 Loading Model...")
print(f"📁 Model: {MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

def chat_function(message, chat_history=None):
    """Chat function using Qwen2-0.5B"""
    if chat_history is None:
        chat_history = []
    
    # Qwen2 prompt format - simple and clean
    context = "သင်သည် မြန်မာစကားပြောတဲ့ AI စာရေးပါ။ မြန်မာလို ပြောပါ။\n\n"
    
    # Build conversation history - only keep last 2 turns
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    for user_msg, bot_msg in recent_history:
        context += f"သုံးစွဲသူ: {user_msg}\nAI: {bot_msg}\n"
    
    context += f"သုံးစွဲသူ: {message}\nAI: "
    
    # Tokenize with truncation
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
    input_ids_len = inputs.input_ids.shape[1]
    
    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.7,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.5,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract response
    answer_tokens = outputs[0][input_ids_len:]
    decoded_output = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Clean up response
    lines = decoded_output.split('\n')
    clean_answer = lines[0].strip() if lines else decoded_output.strip()
    clean_answer = clean_answer.replace("AI:", "").replace("Assistant:", "").replace("သုံးစွဲသူ:", "").strip()
    
    # Ensure it's in Myanmar script
    if clean_answer and not any('\u1000' <= c <= '\u109F' or '\uAA60' <= c <= '\uAA7F' for c in clean_answer[:10] if c):
        clean_answer = "မေးခွန်းရိုက်ပါ။"
    
    if not clean_answer or len(clean_answer) < 2:
        clean_answer = "ပါးလွှတ်ပါပါ။"
    
    return clean_answer

# --- GRADIO UI ---
with gr.Blocks(
    title="Amkyaw AI V2 - Myanmar Chatbot",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("# 🇲🇲 Amkyaw AI V2 - မြန်မာစကားပြော Chatbot")
    
    chatbot = gr.Chatbot(label="စကားပြောမှတ်တမ်း", height=500)
    msg = gr.Textbox(placeholder="မေးခွန်းတစ်ခုခု ရိုက်နှိပ်ပါ...", container=False, scale=4)

    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        bot_message = chat_function(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    gr.Button("ပို့မည်", variant="primary").click(respond, [msg, chatbot], [msg, chatbot])
    gr.ClearButton([msg, chatbot], value="မှတ်တမ်းဖျက်မည်")

if __name__ == "__main__":
    demo.launch()
    
