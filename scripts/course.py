#!/usr/bin/env python3
"""Staged AutoDL pilot. Run one subcommand at a time; see README and learning checkpoints."""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import (digest_file, parse_answer, read_json, read_jsonl, select_data,
                    training_configs, write_json, write_jsonl)

CODE = Path(__file__).resolve().parents[1]
CFG = read_json(CODE / "configs/course.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def training_env(root):
    """Allow a validated alternative environment without changing experiment parameters."""
    return str(Path(os.environ.get("COURSE_TRAIN_ENV", sys.prefix)).expanduser())


def run_path(root, name):
    need(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)), "Use a simple run ID, e.g. smoke01")
    return root / "runs" / name


def inventory(folder):
    """Capture actual downloaded bytes. A mutable ModelScope master ref is NOT a pinned commit."""
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not any(x.startswith(".") for x in path.relative_to(folder).parts):
            files[str(path.relative_to(folder)).replace("\\", "/")] = {
                "bytes": path.stat().st_size, "sha256": digest_file(path)}
    need("config.json" in files, f"Missing model config: {folder}")
    need(any(x.endswith(".safetensors") for x in files), f"Missing safetensors weights: {folder}")
    return files


def verify_inventory(folder, files):
    for name, expected in files.items():
        path = folder / name
        need(path.is_file() and digest_file(path) == expected["sha256"], f"Model file changed/missing: {path}")


def data_ready(root):
    manifest = read_json(root / "data/manifest.json")
    need(manifest["config_sha256"] == digest_file(CODE / "configs/course.json"),
         "Configuration changed since data preparation. Use a new COURSE_DATA_ROOT.")
    for name, sha in manifest["files"].items():
        need(digest_file(root / "data" / name) == sha, f"Prepared data changed: {name}")
    return manifest


def model_ready(root, role):
    manifest = read_json(root / "models" / (role + "-manifest.json"))
    verify_inventory(root / "models" / role, manifest["files"])
    return manifest


def preflight(root, args):
    import torch
    need(sys.version_info[:2] >= (3, 11), "LLaMA-Factory profile requires Python >=3.11; preferred image is Python 3.12")
    need(torch.cuda.is_available(), "CUDA unavailable")
    need(torch.cuda.device_count() == 1, "This profile requires one visible GPU; set CUDA_VISIBLE_DEVICES first")
    need(torch.cuda.is_bf16_supported(), "This profile requires BF16 support")
    need(torch.cuda.get_device_properties(0).total_memory >= 22 * 2**30,
         "This unquantized teacher profile targets a 24 GB GPU; stop and revise for smaller GPUs")
    root.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(["nvidia-smi"], text=True)
    print(gpu)
    report = {"time": now(), "python": sys.version, "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "disk_free_gib": shutil.disk_usage(root).free / 2**30, "nvidia_smi": gpu}
    write_json(root / "environment" / (f"preflight-{time.time_ns()}.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("PAUSE P1: confirm no other GPU jobs; budget teacher/student weights, dependencies, caches and scratch space on the actual disk.")


def download_models(root, args):
    from modelscope import snapshot_download
    for role in ("teacher", "student"):
        manifest_path = root / "models" / (role + "-manifest.json")
        if manifest_path.exists():
            manifest = model_ready(root, role)
            need(manifest["model_id"] == CFG[role], "Model ID changed; use another data root")
            print("Reusing verified model:", role)
            continue
        folder = root / "models" / role
        started = time.perf_counter()
        snapshot_download(CFG[role], revision=CFG["modelscope_revision"], local_dir=str(folder))
        write_json(manifest_path, {"time": now(), "source": "modelscope", "model_id": CFG[role],
                                  "requested_revision": CFG["modelscope_revision"],
                                  "revision_is_immutable": False,
                                  "seconds": time.perf_counter()-started, "files": inventory(folder)})
    print("PAUSE P2: locate weights/configs; hashes preserve actual artifacts. Keep these snapshots for reproducibility.")


def prepare_data(root, args):
    if (root / "data/manifest.json").exists():
        data_ready(root)
        print("Reusing verified prepared data.")
        return
    dest = root / "data"
    need(not dest.exists(), "Incomplete data directory exists. Inspect logs; use a fresh data root or archive it manually.")
    if args.local_data:
        source = Path(args.local_data).resolve()
        paths = [source / x for x in ("numina.jsonl", "gsm8k_train.jsonl", "gsm8k_test.jsonl")]
        numina, train, test = [read_jsonl(p) for p in paths]
        provenance = {"mode": "user_supplied_jsonl", "files": {str(p): digest_file(p) for p in paths},
                      "note": "Local file licenses and revisions must be supplied in the manual record."}
    else:
        from datasets import load_dataset
        numina = load_dataset(CFG["numina_id"], revision=CFG["numina_revision"], split="train")
        train = list(load_dataset(CFG["gsm8k_id"], "main", revision=CFG["gsm8k_revision"], split="train"))
        test = list(load_dataset(CFG["gsm8k_id"], "main", revision=CFG["gsm8k_revision"], split="test"))
        provenance = {"numina": [CFG["numina_id"], CFG["numina_revision"]],
                      "gsm8k": [CFG["gsm8k_id"], CFG["gsm8k_revision"]],
                      "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co")}
    candidates, dev, tests, report = select_data(numina, train, test, CFG)
    for name, rows in (("candidates", candidates), ("dev", dev), ("test", tests)):
        write_jsonl(dest / (name + ".jsonl"), rows)
    write_json(dest / "selection_report.json", report)
    names = ["candidates.jsonl", "dev.jsonl", "test.jsonl", "selection_report.json"]
    write_json(dest / "manifest.json", {"time": now(), "source": provenance,
                                        "config_sha256": digest_file(CODE / "configs/course.json"),
                                        "files": {n: digest_file(dest/n) for n in names}})
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print("PAUSE P2: inspect selection_report.json. This is conservative screening, not proof of zero semantic leakage.")


def engine(model):
    # EngineCore 在主进程内运行：多进程子进程偶发收尾滞留会触发 supervisor 强制回收（exit 2）
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)
    need(bool(tokenizer.chat_template), "No chat template; stop and inspect the pinned model")
    stops = {tokenizer.eos_token_id}
    if "<|im_end|>" in tokenizer.get_vocab():
        stops.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    stops.discard(None)
    started = time.perf_counter()
    llm = LLM(model=str(model), dtype="bfloat16", max_model_len=CFG["cutoff_len"],
              gpu_memory_utilization=CFG["gpu_memory_utilization"], max_num_seqs=4,
              enforce_eager=True, seed=CFG["seed"], trust_remote_code=False)
    return llm, tokenizer, sorted(stops), SamplingParams, time.perf_counter()-started


def prompts(tokenizer, rows):
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template([
            {"role": "system", "content": CFG["system"]},
            {"role": "user", "content": row["question"]}], tokenize=False, add_generation_prompt=True)
        length = len(tokenizer.encode(text, add_special_tokens=False))
        need(length + CFG["max_new_tokens"] <= CFG["cutoff_len"], f"Prompt too long: {row['id']}")
        rendered.append(text)
    return rendered


def student_length(tokenizer, question, answer):
    # Match the qwen template; verify against v0.9.5 preprocessing in the GPU smoke run.
    text = (f"<|im_start|>system\n{CFG['system']}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n")
    return len(tokenizer.encode(text, add_special_tokens=False))


def generate(root, args):
    from transformers import AutoTokenizer
    data_ready(root)
    model_ready(root, "teacher")
    model_ready(root, "student")
    folder = run_path(root, args.run)
    need(not folder.exists(), "Run ID exists. Preserve it and choose a new ID; no silent overwrite.")
    folder.mkdir(parents=True)
    mode = args.mode
    target, limit = (8, 32) if mode == "smoke" else (500, CFG["candidate_count"])
    write_json(folder / "request.json", {"time": now(), "mode": mode, "target": target,
                                         "max_candidates": limit, "config": CFG})
    candidates = read_jsonl(root / "data/candidates.jsonl")[:limit]
    student_tokenizer = AutoTokenizer.from_pretrained(str(root / "models/student"), trust_remote_code=False)
    llm, tokenizer, stops, SamplingParams, load_seconds = engine(root / "models/teacher")
    params = SamplingParams(temperature=0.7, top_p=0.8, top_k=-1, repetition_penalty=1.0, max_tokens=CFG["max_new_tokens"],
                            seed=CFG["seed"], stop_token_ids=stops)
    accepted = []
    started = time.perf_counter()
    with (folder / "generation_audit.jsonl").open("x", encoding="utf-8") as audit:
        for offset in range(0, len(candidates), 8):
            batch = candidates[offset:offset+8]
            outputs = llm.generate(prompts(tokenizer, batch), params, use_tqdm=True)
            need(len(outputs) == len(batch), "Teacher output count mismatch")
            for row, output in zip(batch, outputs):
                completion = output.outputs[0]
                text = completion.text.strip()
                parsed = parse_answer(text)
                length = student_length(student_tokenizer, row["question"], text)
                reasons = []
                if completion.finish_reason != "stop": reasons.append("unfinished_generation")
                if not parsed["format_ok"]: reasons.append("answer_format")
                if parsed["value"] != row["gold"]: reasons.append("wrong_final_answer")
                if length > CFG["cutoff_len"]: reasons.append("too_long_for_student")
                if len(accepted) >= target: reasons.append("surplus_after_target")
                record = dict(row, teacher_text=text, parsed=parsed, student_tokens=length,
                              finish_reason=completion.finish_reason, rejection_reasons=reasons)
                audit.write(json.dumps(record, ensure_ascii=False) + "\n")
                audit.flush()
                if not reasons:
                    accepted.append({"instruction": row["question"], "input": "", "output": text,
                                     "system": CFG["system"], "source_id": row["id"]})
            print(f"Accepted {len(accepted)}/{target}; examined {min(offset+8, len(candidates))}/{limit}", flush=True)
            if len(accepted) == target:
                break
    need(len(accepted) == target,
         f"Accepted only {len(accepted)}/{target}. STOP P3: inspect generation_audit.jsonl; do not relabel as 500.")
    write_json(folder / "dataset/train.json", accepted)
    write_json(folder / "dataset/dataset_info.json", {"teacher_data": {
        "file_name": "train.json", "formatting": "alpaca", "columns": {
            "prompt": "instruction", "query": "input", "response": "output", "system": "system"}}})
    write_json(folder / "generation_complete.json", {"time": now(), "mode": mode, "accepted": len(accepted),
               "load_seconds": load_seconds, "generate_and_filter_seconds": time.perf_counter()-started,
               "train_sha256": digest_file(folder / "dataset/train.json"),
               "data_manifest_sha256": digest_file(root / "data/manifest.json"),
               "limitation": "Correct final number and intact format do not prove all reasoning steps are correct."})
    print("PAUSE P3: manually read at least 10 retained solutions (all 8 for smoke) plus rejected examples before training.")


def make_config(root, args):
    folder = run_path(root, args.run)
    done = read_json(folder / "generation_complete.json")
    need(digest_file(folder / "dataset/train.json") == done["train_sha256"], "Training data changed; inspect")
    train, export = training_configs(root, folder, CFG, done["mode"])
    # JSON is valid YAML and gives deterministic, readily inspected values.
    write_json(folder / "train_config.yaml", train)
    write_json(folder / "export_config.yaml", export)
    print("Config files created. PAUSE P4: inspect dataset, template, cutoff, LoRA, output paths; JSON syntax is valid YAML.")


def train_or_export(root, args):
    folder = run_path(root, args.run)
    stage = args.command
    data_ready(root)
    done = read_json(folder / "generation_complete.json")
    need(digest_file(folder / "dataset/train.json") == done["train_sha256"], "Training data changed")
    need(not (folder / "frozen.json").exists(), "Run frozen; use a new run for changes")
    target = folder / ("adapter" if stage == "train" else "merged")
    need(not target.exists(), f"{target} exists; do not overwrite. Inspect failure or fork a new run.")
    if stage == "export":
        read_json(folder / "train_complete.json")
        need((folder / "adapter/adapter_config.json").exists(), "Missing adapter")
    config = folder / ("train_config.yaml" if stage == "train" else "export_config.yaml")
    need(config.exists(), "Run make-config first")
    config_sha = digest_file(config)
    env_prefix = training_env(root)
    bin_dir = Path(env_prefix) / ("Scripts" if os.name == "nt" else "bin")
    train_python = str(bin_dir / ("python.exe" if os.name == "nt" else "python"))
    train_cli = str(bin_dir / ("llamafactory-cli.exe" if os.name == "nt" else "llamafactory-cli"))
    need(Path(train_python).is_file() and Path(train_cli).is_file(),
         "Training environment missing Python/llamafactory-cli; install in current environment or set COURSE_TRAIN_ENV")
    subprocess.run([train_python, "-c",
                    "import torch; print('Training torch:', torch.__version__, 'CUDA:', torch.version.cuda); "
                    "assert torch.cuda.is_available(), 'Training CUDA unavailable'; "
                    "assert torch.cuda.is_bf16_supported(), 'Training BF16 unavailable'"], check=True)
    cmd = [train_cli, stage, str(config)]
    started = time.perf_counter()
    print("Executing:", cmd, flush=True)
    with (folder / f"{stage}.log").open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", start_new_session=(os.name == "posix"))
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        except BaseException:
            if os.name == "posix":
                import signal
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait()
            raise
    need(code == 0, f"{stage} exited {code}; inspect {stage}.log, preserve this run")
    need(digest_file(config) == config_sha, "Configuration changed while the stage was running; inspect and preserve this attempt")
    if stage == "train":
        need((target / "adapter_config.json").exists(), "Command returned zero but adapter missing")
        need(any(target.glob("adapter_model.*")), "Adapter weights missing")
    else:
        write_json(folder / "merged_manifest.json", {"files": inventory(target), "time": now()})
    write_json(folder / f"{stage}_complete.json", {"time": now(), "command": cmd,
               "seconds": time.perf_counter()-started, "config_sha256": config_sha})
    print(f"PAUSE P4: {stage} finished. Reload/evaluate the exported model before declaring the workflow passed.")


def fork_run(root, args):
    source, target = run_path(root, args.source_run), run_path(root, args.run)
    read_json(source / "generation_complete.json")
    need(not target.exists(), "Destination run exists")
    target.mkdir(parents=True)
    for name in ("request.json", "generation_complete.json", "generation_audit.jsonl"):
        shutil.copy2(source / name, target / name)
    shutil.copytree(source / "dataset", target / "dataset")
    write_json(target / "fork_source.json", {"source": args.source_run, "time": now(),
                                            "note": "Only generation data reused; training/export/evaluation must be rerun."})
    make_config(root, args)


def frozen_files(root, folder):
    files = [CODE / "configs/course.json", root / "data/manifest.json", root / "data/test.jsonl",
             root / "data/dev.jsonl", root / "models/teacher-manifest.json", root / "models/student-manifest.json",
             folder / "dataset/train.json", folder / "dataset/dataset_info.json", folder / "train_config.yaml",
             folder / "export_config.yaml", folder / "merged_manifest.json"]
    files += sorted((CODE / "scripts").glob("*.py"))
    return {str(p): digest_file(p) for p in files}


def freeze(root, args):
    folder = run_path(root, args.run)
    need(read_json(folder / "generation_complete.json")["mode"] == "pilot500", "Smoke runs do not use final test")
    train_done = read_json(folder / "train_complete.json")
    export_done = read_json(folder / "export_complete.json")
    need(digest_file(folder / "train_config.yaml") == train_done["config_sha256"], "Training config changed after training")
    need(digest_file(folder / "export_config.yaml") == export_done["config_sha256"], "Export config changed after export")
    for role in ("teacher", "before", "student"):
        summary = read_json(folder / "eval_dev" / (role + "_summary.json"))
        need(summary["fingerprint"] == evaluation_fingerprint(root, "dev"),
             "Code/settings/dev data changed since development evaluation; use a new run")
    write_json(folder / "frozen.json", {"time": now(), "files": frozen_files(root, folder),
               "note": "Configuration fixed before final test. New tuning requires a new run and disclosure of test exposure."})
    print("PAUSE P5: configuration frozen. Final test is next; do not tune on final-test scores.")


def evaluation_fingerprint(root, split):
    return {"config_sha256": digest_file(CODE / "configs/course.json"),
            "data_sha256": digest_file(root / "data" / (split + ".jsonl")),
            "code": {name: digest_file(CODE / "scripts" / name) for name in ("course.py", "common.py")}}


class DeviceMemory:
    """Whole-device sampled memory, INCLUDING vLLM cache and other processes. Not minimum VRAM."""
    def __init__(self):
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                result = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                                                 text=True, timeout=5)
                for line in result.strip().splitlines():
                    uuid, value = line.split(",")
                    self.samples.append({"time": now(), "gpu_uuid": uuid.strip(), "used_mib": float(value.strip())})
            except Exception as exc:
                self.errors.append(str(exc))
            self.stop_event.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=7)


def evaluate(root, args):
    data_ready(root)
    folder = run_path(root, args.run)
    done = read_json(folder / "generation_complete.json")
    if args.split == "test":
        frozen = read_json(folder / "frozen.json")
        need(frozen_files(root, folder) == frozen["files"], "Frozen inputs/code changed. Do not evaluate a modified run.")
    role = args.model
    if role == "student":
        read_json(folder / "export_complete.json")
        model = folder / "merged"
        files = read_json(folder / "merged_manifest.json")["files"]
        verify_inventory(model, files)
    else:
        model_role = "student" if role == "before" else "teacher"
        model = root / "models" / model_role
        files = model_ready(root, model_role)["files"]
    dest = folder / ("eval_" + args.split)
    need(not (dest / (role + "_predictions.jsonl")).exists(), "Evaluation outputs exist; preserve them. Fork a new run if needed.")
    rows = read_jsonl(root / "data" / (args.split + ".jsonl"))
    if done["mode"] == "smoke":
        need(args.split == "dev", "Smoke evaluates dev only")
        rows = rows[:5]
    need(bool(rows), "Empty evaluation set")
    with DeviceMemory() as memory:
        llm, tokenizer, stops, SamplingParams, load_seconds = engine(model)
        params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, seed=CFG["seed"], max_tokens=CFG["max_new_tokens"], stop_token_ids=stops)
        # Fixed non-benchmark warm-up; never scored.
        llm.generate(prompts(tokenizer, [{"id": "warmup", "question": "What is 1 plus 1?"}]),
                     SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, seed=CFG["seed"], max_tokens=32, stop_token_ids=stops), use_tqdm=False)
        text_prompts = prompts(tokenizer, rows)
        started = time.perf_counter()
        outputs = llm.generate(text_prompts, params, use_tqdm=True)
        seconds = time.perf_counter() - started
    need(len(outputs) == len(rows), "Evaluation count mismatch")
    records = []
    for row, output in zip(rows, outputs):
        completion = output.outputs[0]
        parsed = parse_answer(completion.text)
        records.append(dict(row, text=completion.text, **parsed, finish_reason=completion.finish_reason,
                            output_tokens=len(completion.token_ids), token_ids=list(completion.token_ids),
                            stop_reason=getattr(completion, "stop_reason", None),
                            correct=parsed["value"] == row["gold"] and completion.finish_reason == "stop"))
    write_jsonl(dest / (role + "_predictions.jsonl"), records)
    write_jsonl(dest / (role + "_memory.jsonl"), memory.samples)
    n = len(records)
    peaks = {}
    for sample in memory.samples:
        key = sample["gpu_uuid"]
        peaks[key] = max(peaks.get(key, 0), sample["used_mib"])
    summary = {"time": now(), "model": role, "split": args.split, "n": n,
               "fingerprint": evaluation_fingerprint(root, args.split),
               "correct": sum(x["correct"] for x in records),
               "accuracy": sum(x["correct"] for x in records)/n,
               "format_rate": sum(x["format_ok"] for x in records)/n,
               "parse_rate": sum(x["value"] is not None for x in records)/n,
               "normal_stop_count": sum(x["finish_reason"] == "stop" for x in records),
               "model_path": str(model),
               "generation_config": read_json(model / "generation_config.json") if (model / "generation_config.json").exists() else None,
               "truncated_rate": sum(x["finish_reason"] == "length" for x in records)/n,
               "load_seconds": load_seconds, "batch_generate_seconds": seconds,
               "output_tokens": sum(x["output_tokens"] for x in records),
               "output_tokens_per_second": sum(x["output_tokens"] for x in records)/seconds,
               "weights_gib": sum(x["bytes"] for name, x in files.items() if name.endswith(".safetensors"))/2**30,
               "sampled_whole_device_peak_mib_by_uuid": peaks, "memory_sampling_errors": memory.errors,
               "memory_scope": "whole device, during engine loading/warmup/eval, includes KV cache; NOT model-only or minimum VRAM",
               "eval_settings": {"max_new_tokens": CFG["max_new_tokens"], "temperature": 0,
                                 "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0, "seed": CFG["seed"], "stop_token_ids": stops,
                                 "max_num_seqs": 4, "enforce_eager": True,
                                 "gpu_memory_utilization": CFG["gpu_memory_utilization"], "dtype": "bfloat16"}}
    write_json(dest / (role + "_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("PAUSE P5/P6: compare raw answers and format errors. Memory includes the engine cache; throughput is not single-question latency.")


def report(root, args):
    folder = run_path(root, args.run)
    summaries = []
    for split in ("dev", "test"):
        for role in ("teacher", "before", "student"):
            path = folder / ("eval_" + split) / (role + "_summary.json")
            if path.exists(): summaries.append(read_json(path))
    need(bool(summaries), "No completed evaluations")
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    fields = ["split", "model", "n", "correct", "accuracy", "format_rate", "parse_rate", "truncated_rate",
              "batch_generate_seconds", "output_tokens_per_second", "weights_gib"]
    with (folder / f"results-{stamp}.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    lines = [f"# Run {args.run}: measured results", "", f"Generated: {now()}", "",
             "| Split | Model | N | Correct | Accuracy | Format | Truncated | Output tok/s | Weights GiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['split']} | {s['model']} | {s['n']} | {s['correct']} | {s['accuracy']:.2%} | "
                     f"{s['format_rate']:.2%} | {s['truncated_rate']:.2%} | {s['output_tokens_per_second']:.2f} | {s['weights_gib']:.3f} |")
    complete = all((folder / "eval_test" / (r+"_summary.json")).exists() for r in ("teacher", "before", "student"))
    lines += ["", f"Three-model final-test outputs complete: {complete}.", "",
              "Observed changes are not prewritten conclusions. Inspect per-question outputs and manual reasoning checks.",
              "Device-memory samples include vLLM KV cache. See per-model summary JSON; do not call these minimum VRAM.",
              "A 300-question pilot is not a full benchmark; no gold-SFT control is included.", "",
              "Manual interpretation, hardware exclusivity and actual run timing: record these in your experiment report."]
    (folder / f"results-{stamp}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


def note(root, args):
    folder = run_path(root, args.run)
    need(folder.exists(), "Run does not exist")
    with (folder / "learning_notes.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now(), "checkpoint": args.checkpoint, "text": args.text}, ensure_ascii=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("COURSE_DATA_ROOT", "/root/distill-work"))
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "download-models", "prepare-data", "generate", "make-config", "train", "export", "fork-run", "freeze", "eval", "report", "note"):
        p = sub.add_parser(command)
        if command not in ("preflight", "download-models", "prepare-data"):
            p.add_argument("--run", required=True)
        if command == "prepare-data": p.add_argument("--local-data")
        if command == "generate": p.add_argument("--mode", choices=["smoke", "pilot500"], required=True)
        if command == "fork-run": p.add_argument("--source-run", required=True)
        if command == "eval":
            p.add_argument("--model", choices=["teacher", "before", "student"], required=True)
            p.add_argument("--split", choices=["dev", "test"], required=True)
        if command == "note":
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--text", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(root / "cache/huggingface"))
    os.environ.setdefault("MODELSCOPE_CACHE", str(root / "cache/modelscope"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    root.mkdir(parents=True, exist_ok=True)
    event = {"start": now(), "command": vars(args), "status": "running"}
    started = time.perf_counter()
    functions = {"preflight": preflight, "download-models": download_models, "prepare-data": prepare_data,
                 "generate": generate, "make-config": make_config, "train": train_or_export,
                 "export": train_or_export, "fork-run": fork_run, "freeze": freeze, "eval": evaluate,
                 "report": report, "note": note}
    try:
        functions[args.command](root, args)
        event["status"] = "complete"
    except BaseException as exc:
        event.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        event.update(end=now(), seconds=time.perf_counter()-started)
        write_json(root / "events" / f"{time.time_ns()}-{args.command}.json", event)
        if os.environ.get("COURSE_STAGE_STATUS"):
            write_json(Path(os.environ["COURSE_STAGE_STATUS"]), event)


if __name__ == "__main__":
    main()
