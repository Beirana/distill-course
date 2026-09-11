import torch
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

print("🚀 开始硬标签蒸馏训练（增强版：2000条数据 + 800步 + LoRA扩容）")

# ================= 1. 加载学生模型 =================
model_name = "./Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

student_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

student_model.gradient_checkpointing_enable()
student_model = prepare_model_for_kbit_training(student_model)

# ================= 2. LoRA 扩容（含 FFN 层） =================
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

# ================= 3. 数据加载 =================
dataset = load_dataset("json", data_files={
    "train": "train_1900.jsonl",
    "validation": "val_100.jsonl"
})

MAX_LEN = 512

def preprocess_function(examples):
    """
    关键技巧：只对答案部分计算 loss。
    问题部分（instruction）的 label 设为 -100，让模型专注生成答案。
    """
    input_ids_list = []
    labels_list = []
    attention_list = []

    for instr, out in zip(examples["instruction"], examples["output"]):
        # 分别编码，避免手动拼接出错
        instr_ids = tokenizer(
            instr + "\n",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN // 2,
        )["input_ids"]

        answer_ids = tokenizer(
            out,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN - len(instr_ids) - 1,
        )["input_ids"]

        # EOS 作为结尾
        eos_id = tokenizer.eos_token_id
        ids = instr_ids + answer_ids + [eos_id]

        # 截断/填充到 MAX_LEN
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        pad_len = MAX_LEN - len(ids)
        attn = [1] * len(ids) + [0] * pad_len
        ids = ids + [tokenizer.pad_token_id] * pad_len

        # labels：问题部分设为 -100
        label = [-100] * len(instr_ids) + answer_ids + [eos_id]
        if len(label) > MAX_LEN:
            label = label[:MAX_LEN]
        label = label + [-100] * (MAX_LEN - len(label))

        input_ids_list.append(ids)
        labels_list.append(label)
        attention_list.append(attn)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_list,
    }

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

print(f"✅ 数据准备完成：训练 {len(tokenized_dataset['train'])} 条，验证 {len(tokenized_dataset['validation'])} 条")

# ================= 4. 训练参数 =================
training_args = TrainingArguments(
    output_dir="./distill_output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,      # 模拟 batch_size = 16
    max_steps=800,
    learning_rate=1e-4,
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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ================= 5. 创建 Trainer =================
trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

print("⏳ 训练开始（GPU 加速，预计 30~60 分钟）...")
trainer.train()

# ================= 6. 保存模型 =================
student_model.save_pretrained("./distill_final_lora")
tokenizer.save_pretrained("./distill_final_lora")
print("✅ 训练完成！模型已保存至 ./distill_final_lora")