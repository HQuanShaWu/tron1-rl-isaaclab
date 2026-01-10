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

# ==============================================================================
# [新增] 鲁棒的指令处理器
# ==============================================================================
def refine_instruction(user_input: str) -> str:
    """
    将自然语言导航指令转化为标准的 Pointing Prompt。
    例如: "Please go to the red ball now!" -> "point the red ball"
    """
    # 1. 基础清洗：转小写，去首尾空格，去标点
    text = user_input.lower().strip().rstrip('.,!?')
    
    # 2. 定义干扰词模式 (动词短语)
    # 覆盖：go to, navigate to, arrive at, move towards, find, locate, etc.
    # 使用 regex 的 ^ 表示只匹配开头，避免误删中间的词
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
            # 提取动词后面的部分：即目标物体
            target_object = text[match.end():].strip()
            found_prefix = True
            break
            
    # 3. 处理一些介词残留 (例如 "go to *the* red ball" -> "red ball" 保留the也无所谓，但"point the..."更通顺)
    # RoboBrain 对 "point the red ball" 或 "point red ball" 都能理解
    
    # 4. 兜底策略：
    # 如果用户只输了 "red ball" (名词)，直接加上 point
    # 如果用户输了 "point the red ball"，保持不变
    
    if target_object.startswith("point "):
        return target_object # 已经是标准格式
    
    # 5. 重组为标准格式
    refined_prompt = f"point {target_object}"
    
    print(f"[NLP] Original: '{user_input}' -> Target: '{target_object}' -> Final: '{refined_prompt}'")
    return refined_prompt

# ==============================================================================
# VLM Class (保持不变)
# ==============================================================================
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
        # ... (这里完全保持你之前的代码不变) ...
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

# --- 通信逻辑 ---
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

        # 1. [关键步骤] 预处理指令
        processed_instruction = refine_instruction(raw_input)

        # 2. Capture
        print(">> Capturing image...")
        resp = send_cmd("CAPTURE")
        if not resp or resp["status"] != "OK":
            print("Capture failed.")
            continue
        
        # 3. Inference (使用处理过的指令)
        img_path = resp["image_path"]
        print(f">> Thinking on {img_path}...")
        
        # 这里传入 processed_instruction (例如 "point the red ball")
        pts = brain.predict_point(processed_instruction, img_path)
        
        if pts:
            u, v = pts[0]
            print(f">> Target Pixel: ({u}, {v})")
            # 4. Navigate
            nav_resp = send_cmd("NAVIGATE", {"u": u, "v": v})
            if nav_resp and nav_resp["status"] == "MOVING":
                print(">> Robot MOVING!")
            else:
                print(f">> Navigation failed: {nav_resp}")
        else:
            print(">> No target found.")

if __name__ == "__main__":
    main()