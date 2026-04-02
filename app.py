#!/usr/bin/env python3
"""
Amkyaw-SpaceV1 - Gradio Web Interface for Myanmar LLM
"""
import os
import yaml
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


# Load configuration
def load_config(config_path: str = "space_config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


config = load_config()
chat_engine = None


def generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_p: float = 0.9):
    """Generate text from prompt"""
    if chat_engine is None:
        return "Model not loaded yet..."
    
    try:
        return chat_engine.generate(prompt, max_new_tokens, temperature, top_p)
    except Exception as e:
        return f"Error: {str(e)}"


def chat(message: str, history: list):
    """Handle chat interaction"""
    if chat_engine is None:
        return "Model not loaded yet..."
    
    conversation = "The following is a conversation with an AI assistant.\n\n"
    for user_msg, assistant_msg in history:
        conversation += f"User: {user_msg}\nAssistant: {assistant_msg}\n"
    conversation += f"User: {message}\nAssistant:"
    
    try:
        response = chat_engine.generate(conversation)
        return response
    except Exception as e:
        return f"Error: {str(e)}"


def load_model_on_startup():
    """Load model during startup"""
    global chat_engine
    
    model_config = config.get("model", {})
    model_name = model_config.get("name", "URajinda/ShweYon-V3-Base")
    device = model_config.get("device", "auto")
    torch_dtype = model_config.get("torch_dtype", "float16")
    
    print(f"Loading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=getattr(torch, torch_dtype, torch.float16)
        )
        
        class ChatEngine:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
            
            def generate(self, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                                  temperature=temperature, top_p=top_p, do_sample=True)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if result.startswith(prompt):
                    result = result[len(prompt):].strip()
                return result
        
        chat_engine = ChatEngine(model, tokenizer)
        print("Model loaded successfully!")
        return "✅ Model loaded successfully!"
    except Exception as e:
        print(f"Error loading model: {e}")
        return f"⚠️ Model not fully loaded: {str(e)}"


def build_interface():
    with gr.Blocks(title=config.get("space", {}).get("title", "Amkyaw Myanmar LLM")) as demo:
        gr.Markdown(f"""
        # {config.get("space", {}).get("emoji", "🇲🇲")} {config.get("space", {}).get("title", "Amkyaw Myanmar LLM")}
        
        {config.get("space", {}).get("description", "Interactive Myanmar Language Model Demo")}
        """)
        
        with gr.Tab("Text Generation"):
            with gr.Row():
                with gr.Column(scale=3):
                    prompt_input = gr.Textbox(label="Prompt / စာသွင်းလိုက်ပါနော်", lines=4, placeholder="မြန်မာစာနဲ့ စာရေးပါစိုးပါနော်")
                with gr.Column(scale=1):
                    max_tokens = gr.Slider(minimum=10, maximum=500, value=200, label="Max New Tokens")
                    temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, label="Temperature")
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top P")
            
            generate_btn = gr.Button("Generate / ဖန်တီးရန်", variant="primary")
            output = gr.Textbox(label="Generated Output / ထွက်လာတဲ့စာ", lines=8)
            
            generate_btn.click(fn=generate, inputs=[prompt_input, max_tokens, temperature, top_p], outputs=output)
        
        with gr.Tab("Chat Mode"):
            chatbot = gr.Chatbot(label="Conversation", height=400)
            
            with gr.Row():
                msg_input = gr.Textbox(label="Message", placeholder="ဟောင်းပါနော်၊ မင်္ဂလာပါ", lines=2, scale=4)
                send_btn = gr.Button("Send / ပို့ရန်", scale=1)
            
            def respond(message, history):
                response = chat(message, history)
                history.append((message, response))
                return "", history
            
            send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
            msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        
        startup_status = load_model_on_startup()
        gr.Markdown(f"**Status**: {startup_status}")
    
    return demo


def main():
    print("Starting Amkyaw-SpaceV1...")
    demo = build_interface()
    
    server_port = int(os.environ.get("PORT", 7860))
    demo.launch(server_port=server_port, server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()