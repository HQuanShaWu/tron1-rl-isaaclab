"""
[Brain] VLM Client for Tron 1 Navigation.
Updated with Robust Prompt Engineering.
"""

import socket
import json
import os
import re
import cv2
import torch
from typing import Union

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Please install: pip install transformers qwen-vl-utils")
    exit()


def refine_instruction(user_input: str) -> str:
    """
    将自然语言导航指令转化为标准的 Pointing Prompt。
    例如: "Please go to the red ball now!" -> "point the red ball"
    """
    text = user_input.lower().strip().rstrip('.,!?')
    
    prefixes = [
        r"^(please\s+)?(could\s+you\s+)?(go\s+to|navigate\s+to|arrive\s+at|move\s+to|reach|walk\s+to|run\s+to|head\s+to)\s+",
        r"^(please\s+)?(could\s+you\s+)?(find|locate|search\s+for|look\s+for|spot|detect|identify)\s+",
        r"^(please\s+)?(i\s+want\s+to\s+go\s+to)\s+"
    ]
    
    target_object = text
    found_prefix = False
    
    for p in prefixes:
        match = re.search(p, text)
        if match:
            target_object = text[match.end():].strip()
            found_prefix = True
            break
    
    if target_object.startswith("point "):
        return target_object
    
    refined_prompt = f"point {target_object}"
    
    print(f"[NLP] Original: '{user_input}' -> Target: '{target_object}' -> Final: '{refined_prompt}'")
    return refined_prompt


class UnifiedInference:
    def __init__(self, model_id="BAAI/RoboBrain2.0-3B", device_map="auto"):
        print(f"[VLM] Loading Model: {model_id} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype="auto", device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.supports_thinking = "3b" not in model_id.lower()
        print(f"[VLM] Ready. Thinking: {self.supports_thinking}")
        
    def predict_point(self, text: str, image_path: str):
        prompt = f"{text}. Your answer should be formatted as a list of tuples..."
        
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": prompt}
        ]}]

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.supports_thinking: text_input += "<think></think><answer>"

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        if self.supports_thinking and "</think>" in output_text:
            answer = output_text.split("</think>")[1].replace("<answer>", "").replace("</answer>", "").strip()
        elif self.supports_thinking:
            answer = output_text.replace("<answer>", "").replace("</answer>", "").strip()
        else:
            answer = output_text
        
        print(f"[VLM] Raw Output: {answer}")
        points = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', answer)
        return [(int(x), int(y)) for x, y in points]

HOST, PORT = '127.0.0.1', 65432

def send_cmd(cmd, data=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            s.sendall(json.dumps({"cmd": cmd, "data": data}).encode('utf-8'))
            return json.loads(s.recv(4096).decode('utf-8'))
        except Exception as e:
            print(f"[Err] Connect failed: {e}")
            return None

def main():
    brain = UnifiedInference()
    print("\n" + "="*40)
    print(" [VLM Client Ready] (Natural Language Supported)")
    print(" Try instructions like:")
    print("  - 'Go to the red ball'")
    print("  - 'Find the big rock'")
    print("  - 'Navigate to the rover'")
    print("="*40 + "\n")

    while True:
        try:
            raw_input = input("Instruction (q to quit) > ").strip()
        except EOFError: break
        
        if raw_input == 'q': break
        if not raw_input: continue

        processed_instruction = refine_instruction(raw_input)

        print(">> Capturing image...")
        resp = send_cmd("CAPTURE")
        if not resp or resp["status"] != "OK":
            print("Capture failed.")
            continue
        
        img_path = resp["image_path"]
        print(f">> Thinking on {img_path}...")
        
        pts = brain.predict_point(processed_instruction, img_path)
        
        if pts:
            u, v = pts[0]
            print(f">> Target Pixel: ({u}, {v})")
            nav_resp = send_cmd("NAVIGATE", {"u": u, "v": v})
            if nav_resp and nav_resp["status"] == "MOVING":
                print(">> Robot MOVING!")
            else:
                print(f">> Navigation failed: {nav_resp}")
        else:
            print(">> No target found.")

if __name__ == "__main__":
    main()