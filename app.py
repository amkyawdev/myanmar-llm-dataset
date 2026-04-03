import os
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
# Using ShweYon-V3-Base - Base model for Myanmar language
MODEL_NAME = "URajinda/ShweYon-V3-Base"

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
    Chat function using ShweYon-V3-Base with Chat Template
    """
    if chat_history is None:
        chat_history = []
    
    # Build messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history
    for user_msg, bot_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    # Apply ShweYon's chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception as e:
        # Fallback to manual template if chat template fails
        prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{message} [/INST]"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids_len = inputs.input_ids.shape[1]
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,  # Increased for better responses
            temperature=0.7,     # Balanced creativity
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,  # Reduce repetition
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the new tokens (response)
    answer_tokens = outputs[0][input_ids_len:]
    decoded_output = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    
    # Clean up response
    clean_answer = (
        decoded_output
        .replace("[SEP]", "")
        .replace("[PREDICTION]", "")
        .replace("[OUT]", "")
        .replace("</s>", "")
        .replace("<s>", "")
        .replace("<|endoftext|>", "")
        .strip()
    )
    
    # Fallback response if empty
    if not clean_answer:
        clean_answer = "နားမလည်ပါဘူးခင်ဗျာ။ တစ်မျိုးပြန်မေးကြည့်ပေးပါ။"
    
    return clean_answer

# --- GRADIO UI ---
with gr.Blocks(
    title="Amkyaw AI V2 - Myanmar Chatbot",
    theme=gr.themes.Soft(
        primary_color="#FFD700",  # Myanmar gold
        secondary_color="#FFEA00",
    )
) as demo:
    
    gr.Markdown("""
    # 🇲🇲 Amkyaw AI V2
    ### မြန်မာစကားပြောတဲ့ AI Chatbot
    
    ---
    💡 မြန်မာလို မေးခွန်းရိုက်ပါ။
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat History / စကားမှတ်ပါ", 
                height=500,
                show_copy_button=True,
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            )
            max_tokens = gr.Slider(
                minimum=64,
                maximum=1024,
                value=512,
                step=64,
                label="Max Tokens",
            )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="မေးခွန်းရိုက်ပါ... (Type your question)",
            container=False,
            scale=4,
            label="Message / မေးခွန်း",
        )
    
    with gr.Row():
        send_btn = gr.Button("📤 ပါးလွှတ်ပါ (Send)", variant="primary")
        clear_btn = gr.Button("🗑️ ရှင်းပါ (Clear)", variant="secondary")
    
    # Chat functions
    def respond(message, chat_history, temp, max_t):
        bot_message = chat_function(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    def clear_chat():
        return []
    
    # Event handlers
    msg.submit(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot, temperature, max_tokens], [msg, chatbot])
    clear_btn.click(clear_chat, [], [chatbot])
    
    gr.Markdown("""
    ---
    ### 📝 Model Info
    - **Model:** ShweYon-V3-Base (URajinda)
    - **Fine-tuning:** Myanmar Chat Dataset
    - **Version:** V2
    
    ### 💡 Tips
    - မြန်မာလို မေးပါ။
    - ရှေ့သမိုင်းကို မှတ်ပါပါ။
    - ပါဝင်ပါတရား ဖြစ်ပါ။
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
