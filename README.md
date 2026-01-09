# Tron 1 RL for Isaac Lab (Fixed for v0.5.x+)

This repository is a fork of [limxdynamics/tron1-rl-isaaclab](https://github.com/limxdynamics/tron1-rl-isaaclab).

## 🛠 Fixes / 修改内容

This branch contains fixes to make the code compatible with **NVIDIA Isaac Lab v0.5.x**.
本项目已适配 NVIDIA Isaac Lab v0.5.x 版本。

**Key Changes / 主要修改:**
1. **rsl_rl/runner (on_policy_runner.py)**: Fixed `step()` and `get_observations()` unpacking issues to match the new Isaac Lab Dictionary/Tuple API. (修复了 step 和观测值返回数量不匹配的问题)
2. **scripts/rsl_rl/play.py**: Updated inference loop to handle dictionary observations from `step()`. (修复了推理/回放脚本中的观测值提取逻辑)
3. **scripts/rsl_rl/train.py**: Replaced deprecated `dump_pickle` with standard python `pickle`. (替换了已弃用的 dump_pickle 函数)

## 🚀 How to Run / 如何运行

1. Install Isaac Lab.
2. clone this project at IsaacLab/ .
2. Install this package:
   ```bash
   cd tron1-rl-isaaclab/exts/bipedal_locomotion
   python -m pip install -e . --no-build-isolation
   # However, this will replace the protobuf-6.33.2+ (installed by isaac_lab 0.5.1+) by protobuf-4.25.8. 
   ```
3. Train:
   ```bash
   python scripts/rsl_rl/train.py --task Isaac-Limx-PF-Blind-Flat-v0 --headless
   ```
4. Play/Visualize:
   ```bash
   python scripts/rsl_rl/play.py --task Isaac-Limx-PF-Blind-Flat-Play-v0
   ```

## 🤖 Tasks
- `Isaac-Limx-PF-Blind-Flat-v0` (Point Foot)
- `Isaac-Limx-WF-Blind-Flat-v0` (Wheel Foot)
- `Isaac-Limx-SF-Blind-Flat-v0` (Sole Foot)
