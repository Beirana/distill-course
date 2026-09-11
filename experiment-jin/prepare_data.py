from datasets import load_dataset
import json
import random

print("🔍 正在加载数据集 AI-MO/NuminaMath-CoT ...")

try:
    full_set = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    # 多取一些用于清洗后仍能保留足够样本
    raw_set = full_set.select(range(2500))

    # ========== 数据清洗：过滤过长/过短的推理过程 ==========
    def is_valid(example):
        sol = example.get('solution', '')
        prob = example.get('problem', '')
        if not sol or not prob:
            return False
        # token 长度粗略用字符数估算，避免引入 tokenizer
        if len(sol) < 100 or len(sol) > 3000:
            return False
        if len(prob) < 20:
            return False
        return True

    cleaned = []
    for i in range(len(raw_set)):
        if is_valid(raw_set[i]):
            cleaned.append(raw_set[i])
        if len(cleaned) >= 2000:
            break

    print(f"✅ 清洗后保留 {len(cleaned)} 条有效样本")

    train_data = []
    for i in range(1900):
        train_data.append({
            "instruction": f"请逐步推理并给出最终答案：\n{cleaned[i]['problem']}",
            "output": cleaned[i]['solution']
        })

    val_data = []
    for i in range(1900, 2000):
        val_data.append({
            "instruction": f"请逐步推理并给出最终答案：\n{cleaned[i]['problem']}",
            "output": cleaned[i]['solution']
        })

    print("✅ 成功加载并清洗真实数据！")

except Exception as e:
    print(f"⚠️ 网络失败（{e}），切换到本地模拟数据模式...")
    # ========== 断网自救：生成模拟数据 ==========
    train_data = []
    val_data = []
    templates = [
        {"q": "小明有{}个苹果，小红有{}个，他们一共有多少个？",
         "a": "解：小明有 {} 个苹果，小红有 {} 个苹果。总数 = {} + {} = {}。答案是：{}。"},
        {"q": "商店有{}支笔，卖出{}支，还剩多少支？",
         "a": "解：初始有 {} 支笔，卖出 {} 支。剩余 = {} - {} = {}。答案是：{}。"},
        {"q": "一个长方形长{}厘米，宽{}厘米，面积是多少？",
         "a": "解：长方形面积 = 长 × 宽 = {} × {} = {}。答案是：{}。"},
        {"q": "一本书有{}页，小军每天读{}页，需要多少天读完？",
         "a": "解：总页数 {} 页，每天读 {} 页。天数 = {} / {} = {}。答案是：{}。"},
    ]
    for i in range(1900):
        t = random.choice(templates)
        a = random.randint(5, 50)
        b = random.randint(1, 20)
        if "长方形" in t["q"]:
            r = a * b
        elif "苹果" in t["q"]:
            r = a + b
        elif "笔" in t["q"]:
            r = a - b if a > b else a + b
        else:
            r = a // b if b > 0 else 0
        train_data.append({
            "instruction": f"请逐步推理并给出最终答案：\n{t['q'].format(a, b)}",
            "output": t["a"].format(a, b, r, r)
        })
    for i in range(100):
        t = random.choice(templates)
        a = random.randint(5, 50)
        b = random.randint(1, 20)
        r = a + b if "苹果" in t["q"] else a * b if "长方形" in t["q"] else a - b
        val_data.append({
            "instruction": f"请逐步推理并给出最终答案：\n{t['q'].format(a, b)}",
            "output": t["a"].format(a, b, r, r)
        })

with open("train_1900.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("val_100.jsonl", "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"📁 训练集: {len(train_data)} 条，验证集: {len(val_data)} 条")
print("\n📋 样例（问题片段）：")
print(train_data[0]['instruction'][:120] + "...")
print("\n📋 样例（答案片段）：")
print(train_data[0]['output'][:200] + "...")