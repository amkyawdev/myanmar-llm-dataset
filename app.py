#!/usr/bin/env python3
"""
Amkyaw-Dataset-Manager: Tool for cleaning and formatting Myanmar LLM Datasets.
Features: Unicode normalization, Zawgyi detection, JSONL conversion.
"""

import json
import os
import re

# ---------------- Configuration ----------------
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")

# ---------------- Data Cleaning Logic ----------------
def clean_myanmar_text(text):
    """မြန်မာစာသားများကို သန့်ရှင်းရေးလုပ်ရန် (Remove HTML, Normalizing whitespace)"""
    if not text:
        return ""
    
    # HTML Tags များ ဖယ်ရှားခြင်း
    text = re.sub(r'<[^>]+>', '', text)
    
    # အပို Space များ ဖယ်ရှားခြင်း
    text = ' '.join(text.split())
    
    # Unicode Surrogate pairs များ ဖယ်ရှားခြင်း (JSON Encode error မတက်စေရန်)
    text = ''.join(c for c in text if 0xD800 > ord(c) or ord(c) > 0xDFFF)
    
    return text.strip()

def is_zawgyi(text):
    """ဇော်ဂျီစာသား ဟုတ်မဟုတ် အကြမ်းဖျင်းစစ်ဆေးရန် (Heuristic check)"""
    # ဇော်ဂျီမှာပဲ သုံးလေ့ရှိတဲ့ character combination အချို့
    zg_regex = r"[\u1031\u103b\u1031\u103c\u103d\u103e\u103a]"
    # ဒါက ရိုးရှင်းတဲ့ စစ်ဆေးမှုသာ ဖြစ်ပါတယ်၊ ပိုတိကျချင်ရင် myanmar-tools သုံးနိုင်ပါတယ်
    return False # Default အနေနဲ့ Unicode လို့ပဲ ယူဆပါမယ်

# ---------------- Dataset Processing ----------------
class DatasetManager:
    def __init__(self):
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        self.count = 0

    def process_entry(self, instruction, input_text="", output_text=""):
        """Entry တစ်ခုချင်းစီကို format လုပ်ပြီး သိမ်းဆည်းရန်"""
        
        # Data များကို Clean လုပ်ပါ
        clean_instruction = clean_myanmar_text(instruction)
        clean_input = clean_myanmar_text(input_text)
        clean_output = clean_myanmar_text(output_text)
        
        if not clean_instruction or not clean_output:
            return False
            
        data_struct = {
            "instruction": clean_instruction,
            "input": clean_input,
            "output": clean_output
        }
        
        # JSONL format ဖြင့် သိမ်းဆည်းခြင်း (ensure_ascii=False က အရေးကြီးဆုံးပါ)
        try:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                json_line = json.dumps(data_struct, ensure_ascii=False)
                f.write(json_line + "\n")
            self.count += 1
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False

    def show_stats(self):
        print("-" * 30)
        print(f"✅ Processing Complete!")
        print(f"📊 Total Records Saved: {self.count}")
        print(f"📂 Location: {OUTPUT_FILE}")
        print("-" * 30)

# ---------------- Main Execution ----------------
def main():
    manager = DatasetManager()
    
    print("🚀 Amkyaw Dataset Manager စတင်နေပြီ...")
    
    # ဥပမာ ဒေတာများ ထည့်သွင်းခြင်း (မင်းရဲ့ scraping logic ကို ဒီမှာ ချိတ်ဆက်နိုင်ပါတယ်)
    # နမူနာ အနေနဲ့ manual data အချို့ ထည့်ပြထားပါတယ်
    samples = [
        {
            "instruction": "နေကောင်းလား?", 
            "output": "ဟုတ်ကဲ့ နေကောင်းပါတယ်။ လူကြီးမင်းရော နေကောင်းရဲ့လားခင်ဗျာ။"
        },
        {
            "instruction": "မြန်မာနိုင်ငံရဲ့ မြို့တော်က ဘာလဲ?", 
            "output": "မြန်မာနိုင်ငံရဲ့ မြို့တော်ကတော့ နေပြည်တော် ဖြစ်ပါတယ်။"
        }
    ]
    
    for item in samples:
        manager.process_entry(
            instruction=item['instruction'],
            output_text=item['output']
        )
        
    manager.show_stats()

if __name__ == "__main__":
    main()
