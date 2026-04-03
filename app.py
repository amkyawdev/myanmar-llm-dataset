import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
# Using ShweYon-V3-Base - Base model for Myanmar language
MODEL_NAME = "URajinda/ShweYon-V3-Base"
DATASET_REPO = "amkyawdev/myanmar-llm-dataset"

# System Prompt - Instructions for the AI assistant
SYSTEM_PROMPT = """သင်သည် မြန်မာစကားပြောတဲ့ AI စာရေးပါ။ မြန်မာလို ပြောပါ။
- ပါဝင်ပါတရား ဖြစ်ပါ။
- ရှင်းလင်းပါ။
- မြန်မာဘာသာစကားနဲ့ ပြောပါ။
- အမှားမလုပ်ပါ။
- သုံးစွဲသူကို ကူညီပါ။"""

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
    """
    Chat function using ShweYon-V3-Base
    """
    if chat_history is None:
        chat_history = []
    
    # Simple formatted prompt - FORCE Myanmar language
    context = """သင်သည် မြန်မာစကားပြောတဲ့ AI စာရေးပါ။ မြန်မာလို ပြောပါ။ အင်္ဂလိုင်းမလုပ်ပါ။
 မေးခွန်း:"""

    # Build conversation history - only keep last 2 turns to stay in context
    recent_history = chat_history[-4:] if len(chat_history) > 4 else chat_history
    for user_msg, bot_msg in recent_history:
        context += f"\nသုံးစွဲသူ: {user_msg}\nAI: {bot_msg}"
    
    context += f"\nသုံးစွဲသူ: {message}\nAI:"

    # Tokenize with truncation
    inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)
    input_ids_len = inputs.input_ids.shape[1]
    
    # Generate with strict parameters to prevent repetition
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.3,
            top_p=0.7,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.5,  # Higher to prevent repeating
            length_penalty=1.2,    # Penalize long responses
            no_repeat_ngram_size=3, # Prevent 3-gram repetition
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract response
    answer_tokens = outputs[0][input_ids_len:]
    decoded_output = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Clean up - extract first line and remove labels
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
) as demo:
    
    gr.Markdown("# 🇲🇲 Amkyaw AI V2 - မြန်မာစကားပြောတဲ့ AI Chatbot")
    
    chatbot = gr.Chatbot(label="Chat History / စကားမှတ်ပါ", height=500)
    msg = gr.Textbox(placeholder="မေးခွန်းရိုက်ပါ...", container=False, scale=4)

    def respond(message, chat_history):
        bot_message = chat_function(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    gr.Button("Send", variant="primary").click(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()
