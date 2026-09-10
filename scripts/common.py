"""Small, testable data and scoring utilities. No GPU dependencies here."""
import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
VALUE = rf"(?:{NUMBER})(?:\s*/\s*(?:{NUMBER}))?"


def number(text):
    text = unicodedata.normalize("NFKC", text).strip().replace("−", "-")
    if not re.fullmatch(VALUE, text):
        return None
    try:
        parts = text.replace(",", "").split("/")
        result = Fraction(Decimal(parts[0]))
        if len(parts) == 2:
            result /= Fraction(Decimal(parts[1]))
        return str(result)
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None


def parse_answer(text):
    """Conservative last-line scoring; never grab an arbitrary intermediate number."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return {"value": None, "format_ok": False, "method": "missing"}
    last = unicodedata.normalize("NFKC", lines[-1]).replace("−", "-")
    patterns = [(rf"Answer:\s*({VALUE})\s*[.。!]?", "answer"),
                (rf"\\boxed\{{\s*({VALUE})\s*\}}\s*[.。!]?", "boxed"),
                (rf"####\s*({VALUE})\s*[.。!]?", "hashes"),
                (rf"({VALUE})\s*[.。!]?", "number_only")]
    for pattern, method in patterns:
        match = re.fullmatch(pattern, last, re.IGNORECASE)
        if match:
            value = number(match.group(1))
            return {"value": value, "format_ok": method == "answer" and value is not None,
                    "method": method}
    return {"value": None, "format_ok": False, "method": "unparsed"}


def gold_answer(text):
    if "####" not in text:
        raise ValueError("GSM8K reference lacks ####; inspect data source")
    result = number(text.rsplit("####", 1)[1].strip())
    if result is None:
        raise ValueError("Unsupported reference answer: " + text[-100:])
    return result


def normalize_question(text):
    # Preserve numbers and math punctuation; removing them could merge different questions.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_data(numina, train, test, cfg):
    """Match Numina questions to human GSM8K train answers; hold out dev BEFORE generation."""
    import random
    from collections import Counter, defaultdict
    train_map = {}
    for i, row in enumerate(train):
        key = normalize_question(row["question"])
        answer = gold_answer(row["answer"])
        if key in train_map and train_map[key]["gold"] != answer:
            raise ValueError("Conflicting reference answers for the same train question")
        train_map[key] = {"gsm8k_train_index": i, "question": row["question"],
                          "gold": answer, "reference_solution": row["answer"]}
    test_keys = {normalize_question(row["question"]) for row in test}
    counts, seen, matched = Counter(), set(), []
    for i, row in enumerate(numina):
        if row.get("source") != "gsm8k":
            continue
        counts["source_gsm8k"] += 1
        question = row["problem"]
        if len(question) >= cfg["max_problem_chars"]:
            counts["too_long"] += 1
            continue
        key = normalize_question(question)
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        if key in test_keys:
            counts["test_exact_overlap_removed"] += 1
            continue
        if key not in train_map:
            counts["not_exactly_matched_to_human_train"] += 1
            continue
        matched.append(dict(train_map[key], id=f"numina-{i}", numina_index=i))
    # Conservative shared contiguous 10-word screening against ALL test questions.
    # This is a heuristic, not semantic decontamination or a pretrained-data guarantee.
    def grams(question):
        words = re.findall(r"\w+|[^\w\s]", normalize_question(question))
        return {tuple(words[i:i+10]) for i in range(max(0, len(words)-9))}
    test_index = defaultdict(set)
    for i, row in enumerate(test):
        for gram in grams(row["question"]):
            test_index[gram].add(i)
    kept, near = [], []
    for row in matched:
        hits = set()
        for gram in grams(row["question"]):
            hits.update(test_index.get(gram, ()))
        if hits:
            near.append({"id": row["id"], "test_indices": sorted(hits)})
        else:
            kept.append(row)
    random.Random(cfg["seed"]).shuffle(kept)
    need = cfg["candidate_count"] + cfg["dev_count"]
    counts.update(matched_before_near_filter=len(matched), near_overlap_removed=len(near),
                  available=len(kept))
    if len(kept) < need:
        raise ValueError(f"Only {len(kept)} matched, deduplicated questions; need {need}. "
                         f"Do not silently switch source or lower the target. Counts: {dict(counts)}")
    if len(test) < cfg["test_count"]:
        raise ValueError("Not enough official test questions")
    dev = kept[:cfg["dev_count"]]
    candidates = kept[cfg["dev_count"]:need]
    indices = list(range(len(test)))
    random.Random(cfg["seed"]).shuffle(indices)
    tests = [{"id": f"gsm8k-test-{i}", "question": test[i]["question"],
              "gold": gold_answer(test[i]["answer"])} for i in indices[:cfg["test_count"]]]
    report = {"counts": dict(counts), "dev": len(dev), "candidates": len(candidates),
              "test": len(tests), "near_overlap_removed": near,
              "limitation": "exact normalized match + shared 10-token screen; no semantic or pretraining audit"}
    return candidates, dev, tests, report


def training_configs(root, run_dir, cfg, mode):
    student = str(root / "models" / "student")
    train = {"model_name_or_path": student, "trust_remote_code": False, "stage": "sft",
             "do_train": True, "finetuning_type": "lora", "lora_rank": 8,
             "lora_alpha": 16, "lora_dropout": 0.0, "lora_target": "all",
             "dataset": "teacher_data", "dataset_dir": str(run_dir / "dataset"),
             "template": "qwen", "cutoff_len": cfg["cutoff_len"],
             "preprocessing_num_workers": 1, "dataloader_num_workers": 0,
             "output_dir": str(run_dir / "adapter"), "logging_steps": 1,
             "save_strategy": "epoch", "plot_loss": True, "report_to": "none",
             "overwrite_output_dir": False, "per_device_train_batch_size": 2,
             "gradient_accumulation_steps": 8, "learning_rate": 0.0001,
             "num_train_epochs": 3.0, "lr_scheduler_type": "cosine", "warmup_ratio": 0.1,
             "bf16": True, "gradient_checkpointing": True, "seed": cfg["seed"],
             "data_seed": cfg["seed"], "train_on_prompt": False}
    if mode == "smoke":
        train.update(max_steps=2, per_device_train_batch_size=1, gradient_accumulation_steps=1)
    export = {"model_name_or_path": student, "adapter_name_or_path": str(run_dir / "adapter"),
              "template": "qwen", "trust_remote_code": False,
              "export_dir": str(run_dir / "merged"), "export_size": 2,
              "export_device": "cpu", "export_legacy_format": False}
    return train, export
