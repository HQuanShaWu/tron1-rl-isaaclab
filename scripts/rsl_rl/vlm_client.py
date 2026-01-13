"""
[Brain] VLM Client for Navigation.
Supports:
1. Natural Language (e.g., "find the red ball")
2. Direct Control (e.g., "stop", "turn left 45")
3. Multi-step Planning (e.g., "go to red ball, then turn left 90, then find the rock")
"""

import socket
import json
import os
import re
import cv2
import torch
import time
from typing import Union

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Please install: pip install transformers qwen-vl-utils")
    exit()

def refine_instruction(user_input: str) -> str:
    text = user_input.lower().strip().rstrip('.,!?')
    
    if text == "stop" or text.startswith("turn"):
        return text

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
    return refined_prompt


def parse_multistep_instruction(long_text: str):
    text = long_text.lower()
    for sep in ['then', 'and', '.', ';']:
        text = text.replace(f" {sep} ", ',')
        text = text.replace(f"{sep} ", ',')
    
    raw_steps = [s.strip() for s in text.split(',') if s.strip()]
    
    steps = []
    for s in raw_steps:
        refined = refine_instruction(s)
        steps.append(refined)
        
    print(f"\n[Planner] Task Queue: {steps}")
    return steps


HOST, PORT = '127.0.0.1', 65432

def send_cmd(cmd, data=None):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            s.sendall(json.dumps({"cmd": cmd, "data": data}).encode('utf-8'))
            return json.loads(s.recv(4096).decode('utf-8'))
        except Exception as e:
            # print(f"[Err] Connect failed: {e}")
            return None

def wait_for_idle():
    print(">> Robot moving...", end="", flush=True)
    while True:
        resp = send_cmd("GET_STATUS")
        if resp is None:
            print(" [Connection Lost]")
            break
        
        if resp.get("status") == "IDLE":
            print(" Done.")
            break
        
        print(".", end="", flush=True)
        time.sleep(0.5)


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
        
        print(f"[VLM] Output: {answer}")
        points = re.findall(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', answer)
        return [(int(x), int(y)) for x, y in points]


def main():
    brain = UnifiedInference()
    print("\n" + "="*50)
    print(" [Client Ready] Supported Multi-step Instructions:")
    print(" e.g. 'Find the red ball, then turn left 90, then go to the rock'")
    print("="*50 + "\n")

    while True:
        try:
            raw_input = input("\n[USER] Instruction > ").strip()
        except EOFError: break
        if raw_input == 'q': break
        if not raw_input: continue

        task_queue = parse_multistep_instruction(raw_input)

        for i, sub_task in enumerate(task_queue):
            print(f"\n--- [Step {i+1}/{len(task_queue)}] Executing: '{sub_task}' ---")
            
            if sub_task == "stop":
                send_cmd("STOP")
                print(">> STOP command sent.")
                import time
                time.sleep(5.0) 
                print(">> Robot stabilized. Proceeding to next task.")
                continue
            
            elif sub_task.startswith("turn"):
                direction = 1.0 if "left" in sub_task else -1.0
                angle_match = re.search(r"(\d+)", sub_task)
                angle = float(angle_match.group(1)) if angle_match else 90.0
                
                print(f">> Turning {'Left' if direction > 0 else 'Right'} {angle} deg...")
                send_cmd("TURN", {"direction": direction, "angle": angle})
                
                wait_for_idle()
                continue

            resp = send_cmd("CAPTURE")
            if not resp or resp["status"] != "OK": 
                print("[Err] Capture failed. Skipping step.")
                continue
            
            img_path = resp["image_path"]

            pts = brain.predict_point(sub_task, img_path)

            if pts:
                u, v = pts[0]
                print(f">> Target identified at ({u}, {v})")
                nav_resp = send_cmd("NAVIGATE", {"u": u, "v": v})
                
                if nav_resp and nav_resp["status"] == "MOVING":
                    wait_for_idle()
                else:
                    print(f"[Err] Navigation rejected: {nav_resp}")
            else:
                print(f"[Warn] VLM could not find target for '{sub_task}'. Skipping.")
        
        print("\n[System] All tasks completed.")

if __name__ == "__main__":
    main()