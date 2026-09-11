import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

print("📊 开始评估蒸馏效果（修复版：独立加载 + 统一 prompt）...")

# ================= 配置参数 =================
NUM_TEST_SAMPLES = 50
BATCH_SIZE = 4
PROMPT_PREFIX = "请逐步推理并给出最终答案：\n"

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model_path = "./Qwen2.5-0.5B-Instruct"

# ================= 1. 加载两份独立模型 =================

# ① 蒸馏前模型（纯净，绝不注入 LoRA）
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_tokenizer.pad_token = base_tokenizer.eos_token
base_tokenizer.padding_side = "left"

# ② 蒸馏后模型（独立加载一份基础模型，再注入 LoRA）
# ⚠️ 关键：必须重新调用一次 from_pretrained，得到一份全新的模型！
distill_base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
distill_model = PeftModel.from_pretrained(distill_base, "./distill_final_lora")
distill_tokenizer = AutoTokenizer.from_pretrained("./distill_final_lora")
distill_tokenizer.pad_token = distill_tokenizer.eos_token
distill_tokenizer.padding_side = "left"

# 简单验证：两份模型不是同一个对象
print(f"✅ base_model 类型: {type(base_model).__name__}")
print(f"✅ distill_model 类型: {type(distill_model).__name__}")

# ================= 2. 加载测试数据 =================
print(f"📥 加载 {NUM_TEST_SAMPLES} 条测试数据...")
test_data = load_dataset("meta-math/MetaMathQA", split="train").select(range(NUM_TEST_SAMPLES))
print(f"✅ 加载完成，共 {len(test_data)} 条")

# ================= 3. 评估函数 =================
def evaluate_and_save(model, tokenizer, test_data, model_name, batch_size=4):
    results = []
    total_time = 0.0
    valid_count = 0

    torch.cuda.reset_peak_memory_stats()

    for i in range(0, len(test_data), batch_size):
        batch = test_data.select(range(i, min(i + batch_size, len(test_data))))
        prompts = []
        batch_samples = []
        for sample in batch:
            # ✅ 使用与训练一致的 prompt 格式
            prompts.append(f"{PROMPT_PREFIX}{sample['query']}")
            batch_samples.append(sample)

        if not prompts:
            continue

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start_time
        total_time += elapsed

        for j, sample in enumerate(batch_samples):
            # 只保留新生成的部分（去掉 prompt）
            input_len = inputs.input_ids[j].shape[0]
            generated_ids = outputs[j][input_len:]
            model_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

            valid_count += 1
            results.append({
                "query": sample['query'],
                "type": sample.get('type', 'unknown'),
                "true_answer": sample.get('response', ''),
                "model_output": model_output,
            })

    avg_time = total_time / valid_count if valid_count > 0 else 0.0
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    filename = f"responses_{model_name}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 回答已保存至 {filename}（{len(results)}条）")

    return results, avg_time, peak_mem

# ================= 4. 执行评估 =================
print(f"\n⏳ 评估蒸馏前模型（{NUM_TEST_SAMPLES}条）...")
responses_before, time_before, mem_before = evaluate_and_save(
    base_model, base_tokenizer, test_data, "before", BATCH_SIZE
)

print(f"\n⏳ 评估蒸馏后模型（{NUM_TEST_SAMPLES}条）...")
responses_after, time_after, mem_after = evaluate_and_save(
    distill_model, distill_tokenizer, test_data, "after", BATCH_SIZE
)

# ================= 5. 打印客观指标 =================
print("\n" + "="*60)
print("📊 客观指标对比")
print("-"*60)
print(f"平均推理时间（秒/题） | 蒸馏前: {time_before:.2f} | 蒸馏后: {time_after:.2f} | 变化: {time_after-time_before:+.2f}")
print(f"峰值显存占用（GB）    | 蒸馏前: {mem_before:.2f} | 蒸馏后: {mem_after:.2f} | 变化: {mem_after-mem_before:+.2f}")
print("="*60)

# ================= 6. 打印样例回答 =================
print("\n📝 样例回答对比（蒸馏前 vs 蒸馏后）")
for i in range(min(3, len(responses_before))):
    print(f"\n--- 样本 {i+1} ---")
    print(f"问题: {responses_before[i]['query'][:80]}...")
    print(f"标准答案: {responses_before[i]['true_answer'][:100]}...")
    print(f"蒸馏前回答: {responses_before[i]['model_output'][:200]}...")
    print(f"蒸馏后回答: {responses_after[i]['model_output'][:200]}...")

print("\n✅ 评估全部完成！请打开 responses_before.json 和 responses_after.json 人工核对答案。")