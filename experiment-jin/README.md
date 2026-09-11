# 同伴实验：本地 Windows 单机硬标签蒸馏（Qwen2.5-1.5B → 0.5B）

本目录是同一蒸馏课题的**另一条独立实现路线**，与仓库主包（根目录）的云端方案互补：不用 LLaMA-Factory，不用 vLLM，全部用 `transformers + peft + Trainer` 手写，在本地 Windows 单机上跑通"准备数据 → 教师生成 → QLoRA 训练 → 前后对比评测"闭环。

## 与主包的差异

| | 主包（仓库根目录） | 本目录 |
|---|---|---|
| 运行环境 | AutoDL Linux 云实例（RTX 4090） | 本地 Windows + PowerShell |
| 训练框架 | LLaMA-Factory 0.9.5 | 纯 transformers + PEFT（Trainer 手写） |
| 教师模型 | Qwen2.5-7B-Instruct（vLLM 推理） | Qwen2.5-1.5B-Instruct（transformers 批量推理） |
| 学生训练 | LoRA（不量化） | QLoRA 4bit NF4 double-quant |
| 数据规模 | 500 条教师生成 + 五关筛选 | NuminaMath-CoT 清洗 2000 条（1900 训练 + 100 验证） |
| 评测 | GSM8K dev100 / test300，三方自动判分 | MetaMathQA 前 50 条，蒸馏前后输出对比 |

## 文件说明

| 文件 | 用途 |
|---|---|
| `prepare_data.py` | 从 AI-MO/NuminaMath-CoT 取前 2500 条，按推理长度清洗后产出 `train_1900.jsonl` / `val_100.jsonl` |
| `train_distill.py` | 训练入口一：直接用数据集自带的 solution 作监督信号（硬标签）训练学生 |
| `train_distill_teacher.py` | 训练入口二：先用 1.5B 教师批量重生成 CoT（`train_1900_distilled.jsonl`，已存在则跳过），再用教师输出训练学生 |
| `evaluate.py` | 加载蒸馏前后两份独立模型（4bit），MetaMathQA 50 条同 prompt 对比；默认对比 `train_distill.py` 的输出目录 |
| `命令.txt` | 作者记录的完整运行步骤（Windows PowerShell） |
| `requirements.txt` | 依赖清单（torch / transformers / peft / bitsandbytes 等） |

学生端关键设置：QLoRA r=32、α=64、dropout=0.05，目标模块覆盖 attention（q/k/v/o）与 FFN（gate/up/down）；800 步、有效 batch 16、lr 1e-4 cosine；instruction 部分 label 置 -100，只对答案计 loss（与主包 `train_on_prompt: false` 同思路）。

## 运行方式

在本目录下按 `命令.txt` 逐条执行：全局装 modelscope 下载两个模型 → 建 `kd_env` 虚拟环境装依赖 → 设 `HF_ENDPOINT=https://hf-mirror.com` → `prepare_data.py` → 二选一运行训练入口 → `evaluate.py`。GPU 不可用时按其注释换装对应 CUDA 版 torch。

## 边界说明

本目录按作者交付原样收录，是独立实现：**未纳入根目录 `bundle-manifest.json` 完整性清单，也未经过主包的验证链**；数据、模型与产物文件（jsonl、权重目录）不随仓库分发，由使用者按脚本自行生成。

## 作者

本部分由 **金正一**（[@JIN-Zhy](https://github.com/JIN-Zhy)）负责设计与实现。

项目整体由 **齐睿**（[@Beirana](https://github.com/Beirana)）维护。
