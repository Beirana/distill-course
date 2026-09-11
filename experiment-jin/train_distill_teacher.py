import os
import gc
import json
import time
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm

# ================== 全局配置 ==================
TEACHER_MODEL = "./Qwen2.5-1.5B-Instruct"
STUDENT_MODEL = "./Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "train_1900.jsonl"
VAL_FILE = "val_100.jsonl"
DISTILL_FILE = "train_1900_distilled.jsonl"   # 教师输出保存的位置
OUTPUT_DIR = "./distill_final_lora_teacher"

PROMPT_PREFIX = "请逐步推理并给出最终答案：\n"

# 教师推理参数
TEACHER_BATCH_SIZE = 8
TEACHER_MAX_NEW_TOKENS = 512
TEACHER_MAX_INPUT_LEN = 256

# 学生训练参数
STUDENT_MAX_LEN = 512
STUDENT_BATCH_SIZE = 2
STUDENT_GRAD_ACCUM = 8
STUDENT_MAX_STEPS = 800
STUDENT_LR = 1e-4

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# ==================================================
# 第一步：教师模型批量推理，生成蒸馏数据
# ==================================================
def generate_teacher_outputs():
    print("\n" + "=" * 60)
    print("📌 第一步：教师模型批量推理，生成蒸馏数据")
    print("=" * 60)

    # 若文件已存在，跳过（避免重复推理）
    if os.path.exists(DISTILL_FILE):
        print(f"✅ 检测到已有蒸馏数据 {DISTILL_FILE}，跳过教师推理。")
        print("   如需重新生成，请删除该文件后重新运行。")
        return

    # 加载教师模型（显存占用约1.2GB）
    print(f"📥 加载教师模型 {TEACHER_MODEL} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    teacher.eval()
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"

    # 加载训练数据（只需要问题）
    print(f"📥 加载训练数据 {TRAIN_FILE} ...")
    train_data = load_dataset("json", data_files=TRAIN_FILE, split="train")
    print(f"✅ 共 {len(train_data)} 条待推理")

    # 批量推理
    teacher_outputs = []
    start_time = time.time()

    for i in tqdm(range(0, len(train_data), TEACHER_BATCH_SIZE), desc="教师推理"):
        batch = train_data.select(range(i, min(i + TEACHER_BATCH_SIZE, len(train_data))))
        prompts = [f"{PROMPT_PREFIX}{s['instruction']}" for s in batch]

        inputs = teacher_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=TEACHER_MAX_INPUT_LEN,
        ).to("cuda")

        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=TEACHER_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=teacher_tokenizer.eos_token_id,
            )

        # 解码时只保留新生成的部分
        for j, s in enumerate(batch):
            input_len = inputs.input_ids[j].shape[0]
            generated = outputs[j][input_len:]
            teacher_answer = teacher_tokenizer.decode(generated, skip_special_tokens=True)
            teacher_outputs.append({
                "instruction": s["instruction"],
                "teacher_output": teacher_answer,
            })

    elapsed = time.time() - start_time
    print(f"✅ 教师推理完成，用时 {elapsed/60:.1f} 分钟")

    # 保存蒸馏数据
    with open(DISTILL_FILE, "w", encoding="utf-8") as f:
        for item in teacher_outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"💾 蒸馏数据已保存至 {DISTILL_FILE}（{len(teacher_outputs)} 条）")

    # 彻底释放教师模型显存
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 教师模型已卸载，显存已释放")


# ==================================================
# 第二步：学生模型训练（用教师输出作为监督）
# ==================================================
def train_student():
    print("\n" + "=" * 60)
    print("📌 第二步：学生模型训练（使用教师生成的蒸馏数据）")
    print("=" * 60)

    # 加载学生模型
    print(f"📥 加载学生模型 {STUDENT_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.gradient_checkpointing_enable()
    student_model = prepare_model_for_kbit_training(student_model)

    # LoRA 配置（扩容版）
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.print_trainable_parameters()

    # 加载蒸馏数据（训练集用教师输出，验证集用原始答案）
    dataset = load_dataset("json", data_files={
        "train": DISTILL_FILE,
        "validation": VAL_FILE,
    })

    def preprocess_train(examples):
        """训练集：用 teacher_output 作为监督信号"""
        input_ids_list, labels_list, attention_list = [], [], []
        for instr, out in zip(examples["instruction"], examples["teacher_output"]):
            instr_ids = tokenizer(
                instr + "\n", add_special_tokens=False,
                truncation=True, max_length=STUDENT_MAX_LEN // 2,
            )["input_ids"]
            answer_ids = tokenizer(
                out, add_special_tokens=False,
                truncation=True, max_length=STUDENT_MAX_LEN - len(instr_ids) - 1,
            )["input_ids"]

            eos_id = tokenizer.eos_token_id
            ids = instr_ids + answer_ids + [eos_id]
            if len(ids) > STUDENT_MAX_LEN:
                ids = ids[:STUDENT_MAX_LEN]
            pad_len = STUDENT_MAX_LEN - len(ids)
            attn = [1] * len(ids) + [0] * pad_len
            ids = ids + [tokenizer.pad_token_id] * pad_len

            label = [-100] * len(instr_ids) + answer_ids + [eos_id]
            if len(label) > STUDENT_MAX_LEN:
                label = label[:STUDENT_MAX_LEN]
            label = label + [-100] * (STUDENT_MAX_LEN - len(label))

            input_ids_list.append(ids)
            labels_list.append(label)
            attention_list.append(attn)
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_list}

    def preprocess_val(examples):
        """验证集：用原始 output 作为监督信号"""
        input_ids_list, labels_list, attention_list = [], [], []
        for instr, out in zip(examples["instruction"], examples["output"]):
            instr_ids = tokenizer(
                instr + "\n", add_special_tokens=False,
                truncation=True, max_length=STUDENT_MAX_LEN // 2,
            )["input_ids"]
            answer_ids = tokenizer(
                out, add_special_tokens=False,
                truncation=True, max_length=STUDENT_MAX_LEN - len(instr_ids) - 1,
            )["input_ids"]

            eos_id = tokenizer.eos_token_id
            ids = instr_ids + answer_ids + [eos_id]
            if len(ids) > STUDENT_MAX_LEN:
                ids = ids[:STUDENT_MAX_LEN]
            pad_len = STUDENT_MAX_LEN - len(ids)
            attn = [1] * len(ids) + [0] * pad_len
            ids = ids + [tokenizer.pad_token_id] * pad_len

            label = [-100] * len(instr_ids) + answer_ids + [eos_id]
            if len(label) > STUDENT_MAX_LEN:
                label = label[:STUDENT_MAX_LEN]
            label = label + [-100] * (STUDENT_MAX_LEN - len(label))

            input_ids_list.append(ids)
            labels_list.append(label)
            attention_list.append(attn)
        return {"input_ids": input_ids_list, "labels": labels_list, "attention_mask": attention_list}

    print("🔧 正在预处理训练集 ...")
    train_tok = dataset["train"].map(
        preprocess_train, batched=True,
        remove_columns=dataset["train"].column_names,
    )
    print("🔧 正在预处理验证集 ...")
    val_tok = dataset["validation"].map(
        preprocess_val, batched=True,
        remove_columns=dataset["validation"].column_names,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./distill_output_teacher",
        per_device_train_batch_size=STUDENT_BATCH_SIZE,
        per_device_eval_batch_size=STUDENT_BATCH_SIZE,
        gradient_accumulation_steps=STUDENT_GRAD_ACCUM,
        max_steps=STUDENT_MAX_STEPS,
        learning_rate=STUDENT_LR,
        warmup_steps=100,
        logging_steps=20,
        eval_steps=100,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        report_to="none",
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    print("⏳ 学生训练开始 ...")
    trainer.train()

    # 保存
    student_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 学生模型已保存至 {OUTPUT_DIR}")


# ==================================================
# 主流程
# ==================================================
if __name__ == "__main__":
    t0 = time.time()

    # 第一步：教师推理
    generate_teacher_outputs()

    # 第二步：学生训练
    train_student()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"🎉 全部完成！总耗时 {elapsed/60:.1f} 分钟")
    print(f"   蒸馏数据: {DISTILL_FILE}")
    print(f"   学生模型: {OUTPUT_DIR}")
    print("=" * 60)