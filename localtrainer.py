#!/usr/bin/env python3
"""
LocalTrainer - Program تدريب نماذج LLM محلية
يُشغّل عبر: python localtrainer.py <command> [options]

الأوامر:
  list                    عرض كل النماذج على القرص
  inspect <model_path>    فحص بنية النموذج (الطبقات، المعاملات، الحجم)
  backup <model_path>     نسخ احتياطي كامل قبل التدريب
  train                   تدريب QLoRA مع Unsloth
  test                    اختبار النموذج (inference)
  export                  تصدير لـ GGUF + Ollama
  check                   فحص الجهاز والـ VRAM

مُبنية على: Unsloth + TRL + PEFT + bitsandbytes
المستوى: GTX 1080 Ti 11GB | 28 cores | 15.5GB RAM
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ─── Constants ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
BACKUPS_DIR = BASE_DIR / "backups"
LOGS_DIR = BASE_DIR / "logs"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


# ═══════════════════════════════════════════════════════════════════════════
#  GPU Detection (borrowed from NTTuner)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GPUInfo:
    has_gpu: bool = False
    gpu_name: str = "None"
    gpu_memory_gb: float = 0.0
    backend: str = "cpu"
    cuda_version: str = ""
    bf16_supported: bool = False


def detect_gpu() -> GPUInfo:
    info = GPUInfo()
    try:
        import torch
        if torch.cuda.is_available():
            info.has_gpu = True
            info.backend = "cuda"
            info.gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info.gpu_memory_gb = props.total_memory / (1024 ** 3)
            info.cuda_version = torch.version.cuda or ""
            # Pascal GPUs (compute capability 6.x) don't support native BF16
            cc = props.major * 10 + props.minor
            info.bf16_supported = torch.cuda.is_bf16_supported() and cc >= 70
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.has_gpu = True
            info.backend = "mps"
            info.gpu_name = "Apple Metal"
            info.gpu_memory_gb = 16.0
    except ImportError:
        pass
    return info


def get_gpu_temp() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


def get_gpu_vram_used_mb() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip()
        return int(out.split("\n")[0])
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def size_str(path: Path) -> str:
    if not path.exists():
        return "0B"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total < 1024:
        return f"{total}B"
    elif total < 1024**2:
        return f"{total/1024:.1f}KB"
    elif total < 1024**3:
        return f"{total/1024**2:.1f}MB"
    else:
        return f"{total/1024**3:.1f}GB"


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_ok(msg: str):
    print(f"  [OK] {msg}")


def print_err(msg: str):
    print(f"  [ERR] {msg}")


def print_info(msg: str):
    print(f"  [..] {msg}")


def ensure_dirs():
    for d in [DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, BACKUPS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: check - فحص الجهاز
# ═══════════════════════════════════════════════════════════════════════════

def cmd_check(args):
    print_header("فحص الجهاز")

    gpu = detect_gpu()
    print(f"  GPU:       {gpu.gpu_name}")
    print(f"  VRAM:      {gpu.gpu_memory_gb:.1f} GB")
    print(f"  Backend:   {gpu.backend}")
    print(f"  CUDA:      {gpu.cuda_version}")
    print(f"  BF16:      {'Yes' if gpu.bf16_supported else 'No'}")
    print(f"  Temp:      {get_gpu_temp()}C")

    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  RAM:       {ram.total/1e9:.1f} GB ({ram.available/1e9:.1f} GB free)")
        print(f"  CPU:       {psutil.cpu_count()} cores")
    except ImportError:
        print(f"  RAM:       (psutil not installed)")

    print()
    libs = {}
    for name in ["torch", "transformers", "peft", "trl", "datasets",
                 "bitsandbytes", "unsloth", "accelerate"]:
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "ok")
        except ImportError:
            libs[name] = "MISSING"

    print("  المكتبات:")
    for k, v in libs.items():
        status = "OK" if v != "MISSING" else "MISSING"
        print(f"    {k:20s} {v}")

    if gpu.has_gpu:
        vram_used = get_gpu_vram_used_mb()
        vram_total = int(gpu.gpu_memory_gb * 1024)
        print(f"\n  VRAM Usage: {vram_used}/{vram_total} MB ({vram_used/vram_total*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: list - عرض النماذج
# ═══════════════════════════════════════════════════════════════════════════

def cmd_list(args):
    print_header("النماذج المتاحة")

    # Scan directories
    search_dirs = []
    if args.path:
        search_dirs.append(Path(args.path))
    else:
        search_dirs.append(MODELS_DIR)
        search_dirs.append(BASE_DIR.parent / "data" / "models" / "shared" / "base")
        search_dirs.append(BASE_DIR.parent / "data" / "models" / "shared" / "trained")
        # Also scan common locations
        for d in [Path.home() / ".ollama" / "models"]:
            if d.exists():
                search_dirs.append(d)

    found = {"base": [], "trained": [], "adapter": [], "other": []}

    for scan_dir in search_dirs:
        if not scan_dir.exists():
            continue
        for p in sorted(scan_dir.iterdir()):
            if not p.is_dir():
                continue

            config_file = p / "config.json"
            adapter_file = p / "adapter_config.json"
            safetensors = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
            gguf_files = list(p.glob("*.gguf"))

            info = {
                "name": p.name,
                "path": str(p),
                "size": size_str(p),
            }

            if adapter_file.exists():
                found["adapter"].append(info)
            elif config_file.exists() and safetensors:
                try:
                    cfg = json.loads(config_file.read_text())
                    info["type"] = cfg.get("model_type", "?")
                    info["layers"] = cfg.get("num_hidden_layers", "?")
                    info["hidden"] = cfg.get("hidden_size", "?")
                    info["vocab"] = cfg.get("vocab_size", "?")
                except Exception:
                    pass
                found["base"].append(info)
            elif gguf_files:
                info["gguf"] = [f.name for f in gguf_files]
                found["other"].append(info)
            elif safetensors:
                found["trained"].append(info)

    # Also check checkpoints directory
    if CHECKPOINTS_DIR.exists():
        for p in sorted(CHECKPOINTS_DIR.iterdir()):
            if p.is_dir():
                adapter_file = p / "adapter_config.json"
                if adapter_file.exists():
                    found["adapter"].append({
                        "name": p.name,
                        "path": str(p),
                        "size": size_str(p),
                    })

    # Also check backups directory
    if BACKUPS_DIR.exists():
        for p in sorted(BACKUPS_DIR.iterdir()):
            if p.is_dir():
                found["trained"].append({
                    "name": f"[BACKUP] {p.name}",
                    "path": str(p),
                    "size": size_str(p),
                })

    for category, items in found.items():
        if not items:
            continue
        label = {"base": "النماذج الأساسية", "trained": "النماذج المدربة",
                 "adapter": "Adapters (LoRA)", "other": "أخرى"}
        print(f"\n  ── {label.get(category, category)} ({len(items)}) ──")
        for item in items:
            name = item["name"]
            path = item["path"]
            size = item["size"]
            extra = ""
            if "type" in item:
                extra = f" | {item['type']} layers={item.get('layers','?')} hidden={item.get('hidden','?')}"
            if "gguf" in item:
                extra = f" | GGUF: {', '.join(item['gguf'])}"
            print(f"    {name:40s} {size:>8s}{extra}")
            print(f"      {path}")

    total = sum(len(v) for v in found.values())
    print(f"\n  الإجمالي: {total} نموذج")


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: inspect - فحص بنية النموذج
# ═══════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print_err(f"المسار غير موجود: {model_path}")
        return

    print_header(f"فحص النموذج: {model_path.name}")
    print(f"  المسار: {model_path}")
    print(f"  الحجم: {size_str(model_path)}")

    # config.json
    config_file = model_path / "config.json"
    if config_file.exists():
        cfg = json.loads(config_file.read_text())
        print(f"\n  ── config.json ──")
        print(f"  Model Type:     {cfg.get('model_type', '?')}")
        print(f"  Architecture:   {cfg.get('architectures', ['?'])}")
        print(f"  Layers:         {cfg.get('num_hidden_layers', '?')}")
        print(f"  Hidden Size:    {cfg.get('hidden_size', '?')}")
        print(f"  Attention Heads:{cfg.get('num_attention_heads', '?')}")
        print(f"  KV Heads:       {cfg.get('num_key_value_heads', '?')}")
        print(f"  Intermediate:   {cfg.get('intermediate_size', '?')}")
        print(f"  Vocab Size:     {cfg.get('vocab_size', '?')}")
        print(f"  Max Position:   {cfg.get('max_position_embeddings', '?')}")
        print(f"  dtype:          {cfg.get('torch_dtype', '?')}")
        print(f"  Rope Theta:     {cfg.get('rope_theta', '?')}")

        # Estimate params
        hidden = cfg.get("hidden_size", 2048)
        layers = cfg.get("num_hidden_layers", 36)
        intermediate = cfg.get("intermediate_size", 11008)
        vocab = cfg.get("vocab_size", 151936)
        params = 12 * layers * hidden**2 + vocab * hidden
        print(f"  ~Params:        {params/1e9:.2f}B")

    # adapter_config.json
    adapter_file = model_path / "adapter_config.json"
    if adapter_file.exists():
        acfg = json.loads(adapter_file.read_text())
        print(f"\n  ── LoRA Adapter ──")
        print(f"  Method:         {acfg.get('peft_type', '?')}")
        print(f"  r:              {acfg.get('r', '?')}")
        print(f"  alpha:          {acfg.get('lora_alpha', '?')}")
        print(f"  dropout:        {acfg.get('lora_dropout', '?')}")
        print(f"  target modules: {acfg.get('target_modules', '?')}")
        print(f"  base model:     {acfg.get('base_model_name_or_path', '?')}")

        # Check for Unsloth
        raw = json.dumps(acfg).lower()
        if "unsloth" in raw:
            print(f"  Framework:      Unsloth")
        else:
            print(f"  Framework:      PEFT/Standard")

    # trainer_state.json
    state_file = model_path / "trainer_state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        print(f"\n  ── Trainer State ──")
        print(f"  Run ID:         {state.get('run_id', '?')}")
        print(f"  Step:           {state.get('step', '?')}")
        print(f"  Epoch:          {state.get('epoch', '?')}")
        print(f"  Final Loss:     {state.get('final_loss', '?')}")

    # Files listing
    print(f"\n  ── الملفات ──")
    for f in sorted(model_path.iterdir()):
        if f.is_file():
            sz = f.stat().st_size
            if sz < 1024:
                sz_str = f"{sz}B"
            elif sz < 1024**2:
                sz_str = f"{sz/1024:.1f}KB"
            else:
                sz_str = f"{sz/1024**2:.1f}MB"
            print(f"    {f.name:45s} {sz_str:>10s}")

    # PEFT summary
    try:
        from peft import PeftModel
        print(f"\n  ── PEFT Info ──")
        print(f"  PEFT available: Yes")
    except ImportError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: backup - نسخ احتياطي
# ═══════════════════════════════════════════════════════════════════════════

def cmd_backup(args):
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print_err(f"المسار غير موجود: {model_path}")
        return

    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{model_path.name}_{timestamp}"
    backup_path = BACKUPS_DIR / backup_name

    print_header(f"نسخ احتياطي: {model_path.name}")
    print_info(f"من: {model_path}")
    print_info(f"إلى: {backup_path}")

    shutil.copytree(model_path, backup_path)

    print_ok(f"تم النسخ: {size_str(backup_path)}")
    print_ok(f"المسار: {backup_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: train - التدريب
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    model_path: str = ""
    data_file: str = ""
    output_dir: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    max_seq_length: int = 2048
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    epochs: int = 3
    warmup_steps: int = 100
    save_every: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "adamw_8bit"
    lr_scheduler: str = "cosine"
    packing: bool = True
    resume_from: str = ""


def cmd_train(args):
    gpu = detect_gpu()
    if not gpu.has_gpu:
        print_err("لا يوجد GPU! التدريب يتطلب GPU")
        return

    print_header("إعداد التدريب")

    # Build config from args
    tc = TrainConfig(
        model_path=args.model or str(MODELS_DIR / "base" / "Qwen2.5-3B-Instruct"),
        data_file=args.data or str(DATA_DIR / "all_categories_cleaned.jsonl"),
        output_dir=args.output or str(CHECKPOINTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_length=args.max_seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        epochs=args.epochs,
        warmup_steps=args.warmup,
        save_every=args.save_every,
        packing=not args.no_packing,
        resume_from=args.resume or "",
    )

    # Validate
    if not Path(tc.model_path).exists():
        print_err(f"النموذج غير موجود: {tc.model_path}")
        return
    if not Path(tc.data_file).exists():
        print_err(f"ملف البيانات غير موجود: {tc.data_file}")
        return

    # Backup first
    print_info("نسخ احتياطي قبل التدريب...")
    backup_name = f"{Path(tc.model_path).name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = BACKUPS_DIR / backup_name
    try:
        shutil.copytree(tc.model_path, backup_path)
        print_ok(f"النسخ الاحتياطي: {backup_path}")
    except Exception as e:
        print_err(f"فشل النسخ الاحتياطي: {e}")
        resp = input("  المتابعة بدون نسخ احتياطي؟ [y/N]: ")
        if resp.lower() != "y":
            return

    effective_batch = tc.batch_size * tc.grad_accum
    print(f"\n  ── إعدادات التدريب ──")
    print(f"  Model:          {Path(tc.model_path).name}")
    print(f"  Data:           {Path(tc.data_file).name}")
    print(f"  Output:         {tc.output_dir}")
    print(f"  LoRA r={tc.lora_r} alpha={tc.lora_alpha} dropout={tc.lora_dropout}")
    print(f"  Max Seq Len:    {tc.max_seq_length}")
    print(f"  Batch:          {tc.batch_size} x {tc.grad_accum} = {effective_batch}")
    print(f"  LR:             {tc.learning_rate}")
    print(f"  Epochs:         {tc.epochs}")
    print(f"  Save every:     {tc.save_every} steps")
    print(f"  Packing:        {tc.packing}")
    print(f"  Optimizer:      {tc.optim}")
    print(f"  Scheduler:      {tc.lr_scheduler}")
    print(f"  GPU:            {gpu.gpu_name} ({gpu.gpu_memory_gb:.1f}GB)")
    print(f"  BF16:           {gpu.bf16_supported}")

    # Count data
    data_count = sum(1 for _ in open(tc.data_file, "r", encoding="utf-8"))
    steps_per_epoch = data_count // effective_batch
    total_steps = steps_per_epoch * tc.epochs
    print(f"  Data:           {data_count} records")
    print(f"  Steps/epoch:    {steps_per_epoch}")
    print(f"  Total steps:    {total_steps}")

    # Build training script
    print_info("بناء سكريبت التدريب...")

    train_script = f'''#!/usr/bin/env python3
"""Auto-generated training script by LocalTrainer"""
import json, os, sys, time
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("  LocalTrainer - بدء التدريب")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# GPU temp check
def check_temp():
    try:
        out = os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits").read().strip()
        return int(out.split("\\n")[0])
    except: return 0

temp = check_temp()
if temp > 70:
    print(f"[WARN] GPU temp {{temp}}C - waiting for cooldown...")
    import time
    while check_temp() > 65:
        time.sleep(30)
    print(f"[OK] Temp now {{check_temp()}}C")

print("\\nLoading model with Unsloth...")
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{tc.model_path}",
    max_seq_length={tc.max_seq_length},
    dtype=None,
    load_in_4bit=True,
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r={tc.lora_r},
    target_modules={json.dumps(TARGET_MODULES)},
    lora_alpha={tc.lora_alpha},
    lora_dropout={tc.lora_dropout},
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {{trainable:,}} / {{total:,}} ({{100*trainable/total:.2f}}%)")

print("\\nLoading dataset...")
records = []
with open("{tc.data_file}", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try: records.append(json.loads(line))
            except: pass
print(f"  {{len(records)}} records loaded")

temp_file = "temp_train_data.jsonl"
with open(temp_file, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\\n")

dataset = load_dataset("json", data_files=temp_file, split="train")

def formatting_func(examples):
    col = "messages" if "messages" in examples else "conversations"
    return {{"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in examples[col]]}}

dataset = dataset.map(formatting_func, batched=True)

# Checkpoint callback
class ManualCheckpoint(TrainerCallback):
    def __init__(self, save_every, output_dir, model, tokenizer):
        self.save_every = save_every
        self.output_dir = Path(output_dir)
        self.model = model
        self.tokenizer = tokenizer
        self.last_save = 0
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step - self.last_save >= self.save_every:
            ckpt_dir = self.output_dir / f"checkpoint-{{state.global_step}}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(ckpt_dir))
            self.tokenizer.save_pretrained(str(ckpt_dir))
            self.last_save = state.global_step
            elapsed = time.time() - self.start_time
            speed = state.global_step / elapsed if elapsed > 0 else 0
            eta = (args.max_steps - state.global_step) / speed if speed > 0 else 0
            temp = check_temp()
            print(f"\\n  [CHECKPOINT] step={{state.global_step}} loss={{kwargs.get('logs',{{}}).get('loss','?')}} temp={{temp}}C ETA={{int(eta//60)}}m{{int(eta%60)}}s")

training_args = TrainingArguments(
    output_dir="{tc.output_dir}",
    per_device_train_batch_size={tc.batch_size},
    gradient_accumulation_steps={tc.grad_accum},
    num_train_epochs={tc.epochs},
    learning_rate={tc.learning_rate},
    fp16={not gpu.bf16_supported},
    bf16={gpu.bf16_supported},
    logging_steps=5,
    save_strategy="no",
    optim="{tc.optim}",
    warmup_steps={tc.warmup_steps},
    lr_scheduler_type="{tc.lr_scheduler}",
    weight_decay={tc.weight_decay},
    max_grad_norm={tc.max_grad_norm},
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={{"use_reentrant": False}},
)

ckpt_callback = ManualCheckpoint({tc.save_every}, "{tc.output_dir}", model, tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length={tc.max_seq_length},
    packing={tc.packing},
    args=training_args,
    callbacks=[ckpt_callback],
)

resume_from = None
resume_dir = Path("{tc.resume_from or tc.output_dir}")
if resume_dir.exists():
    for d in sorted(resume_dir.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            safetensors = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
            if safetensors:
                resume_from = str(d)

if resume_from:
    print(f"\\n  Resuming from: {{resume_from}}")

print("\\n" + "=" * 60)
print("  التدريب يبدأ الآن!")
print("=" * 60)

start = datetime.now()
try:
    trainer.train(resume_from_checkpoint=resume_from)

    print("\\n  Saving LoRA adapter...")
    lora_dir = "{tc.output_dir}/final_lora"
    Path(lora_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("  Merging LoRA into base model...")
    merged_dir = "{tc.output_dir}/merged"
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    elapsed = datetime.now() - start
    print(f"\\n  Training complete! Time: {{elapsed}}")
    print(f"  LoRA adapter: {{lora_dir}}")
    print(f"  Merged model: {{merged_dir}}")

except Exception as e:
    print(f"\\n  [ERROR] Training failed: {{e}}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
'''

    script_path = BASE_DIR / "_train_run.py"
    script_path.write_text(train_script, encoding="utf-8")

    # Run training
    print_info("تشغيل التدريب...")
    print(f"  Script: {script_path}")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        proc.wait()
        if proc.returncode == 0:
            print_ok(f"التدريب اكتمل! المخرجات في: {tc.output_dir}")
        else:
            print_err(f"التدريب فشل (code={proc.returncode})")
    except KeyboardInterrupt:
        print("\n  [STOP] Training interrupted by user")
        proc.terminate()
        proc.wait(timeout=10)
        print_ok("تم الحفظ (إذا كان هناك checkpoint)")
    finally:
        if script_path.exists():
            script_path.unlink()


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: test - اختبار النموذج
# ═══════════════════════════════════════════════════════════════════════════

def cmd_test(args):
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print_err(f"المسار غير موجود: {model_path}")
        return

    print_header(f"اختبار النموذج: {model_path.name}")

    # Detect type
    adapter_config = model_path / "adapter_config.json"
    is_lora = adapter_config.exists()

    print_info("تحميل النموذج...")
    try:
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=args.max_seq_len,
            dtype=None,
            load_in_4bit=True,
        )

        if is_lora:
            print_ok("تم تحميل LoRA adapter على النموذج الأساسي")
        else:
            print_ok("تم تحميل النموذج الكامل")

        # Interactive test
        print(f"\n  اكتب سؤالاً (اكتب 'quit' للخروج):")
        print(f"  {'─'*50}")

        prompt = args.prompt
        if prompt:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            print(f"\n  [{prompt}]")

            FastLanguageModel.for_inference(model)
            start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
            elapsed = time.time() - start
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
            print(f"\n  [{response}]")
            tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            print(f"  ({tokens} tokens in {elapsed:.1f}s = {tokens/elapsed:.1f} tok/s)")
        else:
            FastLanguageModel.for_inference(model)
            while True:
                try:
                    user_input = input("\n  You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

                messages = [{"role": "user", "content": user_input}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)

                start = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
                elapsed = time.time() - start
                response = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
                print(f"  AI> {response}")
                print(f"  ({tokens} tokens in {elapsed:.1ff}s = {tokens/elapsed:.1f} tok/s)")

    except Exception as e:
        print_err(f"خطأ: {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
#  CMD: export - تصدير GGUF + Ollama
# ═══════════════════════════════════════════════════════════════════════════

def cmd_export(args):
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print_err(f"المسار غير موجود: {model_path}")
        return

    print_header(f"تصدير النموذج: {model_path.name}")

    # Check if merged model or LoRA adapter
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        print_err("هذا LoRA adapter فقط! يجب دمج (merge) أولاً:")
        print(f"  المخرجات المدمجة عادة في: {model_path.parent / 'merged'}")
        merged = model_path.parent / "merged"
        if merged.exists():
            print_ok(f"وجدت نموذج مدمج: {merged}")
            model_path = merged
        else:
            print_err("لا يوجد نموذج مدمج. قم بتشغيل التدريب أولاً")
            return

    quant = args.quant or "q4_k_m"
    output_dir = Path(args.output or str(model_path / "gguf"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Model:   {model_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Quant:   {quant}")

    # Try Unsloth GGUF export
    print_info("تصدير GGUF عبر Unsloth...")
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            load_in_4bit=False,
        )

        model.save_pretrained_gguf(
            str(output_dir),
            tokenizer,
            quantization_method=quant.lower(),
        )

        gguf_files = list(output_dir.glob("*.gguf"))
        if gguf_files:
            print_ok(f"GGUF exported: {gguf_files[0]}")
            print_ok(f"  Size: {size_str(gguf_files[0])}")

            # Auto-import to Ollama if requested
            if args.to_ollama:
                model_name = args.ollama_name or model_path.name
                import_to_ollama(gguf_files[0], model_name)
        else:
            print_err("GGUF export produced no files")

    except Exception as e:
        print_err(f"GGUF export failed: {e}")
        import traceback
        traceback.print_exc()


def import_to_ollama(gguf_path: Path, model_name: str):
    print_info(f"Importing to Ollama as '{model_name}'...")

    modelfile = f"""FROM {gguf_path.resolve()}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a helpful AI assistant."
"""
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(modelfile)

    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print_ok(f"Imported to Ollama: {model_name}")
        else:
            print_err(f"Ollama import failed: {result.stderr}")
    except FileNotFoundError:
        print_err("Ollama not installed. Install: curl -fsSL https://ollama.com/install.sh | sh")
    except Exception as e:
        print_err(f"Ollama import error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="LocalTrainer - تدريب نماذج LLM محلية",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة:
  python localtrainer.py check                          فحص الجهاز
  python localtrainer.py list                           عرض النماذج
  python localtrainer.py inspect ./models/Qwen2.5-3B    فحص بنية النموذج
  python localtrainer.py backup ./models/Qwen2.5-3B     نسخ احتياطي
  python localtrainer.py train --model ./models/Qwen2.5-3B-Instruct --data data/all_categories_cleaned.jsonl
  python localtrainer.py test ./checkpoints/run_xxx/final_lora --prompt "كيف أصلح سيرفر؟"
  python localtrainer.py export ./checkpoints/run_xxx/merged --quant q4_k_m --to-ollama
        """
    )

    sub = parser.add_subparsers(dest="command", help="الأمر")

    # check
    sub.add_parser("check", help="فحص الجهاز والـ VRAM والمكتبات")

    # list
    p_list = sub.add_parser("list", help="عرض النماذج على القرص")
    p_list.add_argument("--path", type=str, help="مسار للبحث فيه")

    # inspect
    p_inspect = sub.add_parser("inspect", help="فحص بنية النموذج")
    p_inspect.add_argument("model_path", help="مسار النموذج")

    # backup
    p_backup = sub.add_parser("backup", help="نسخ احتياطي")
    p_backup.add_argument("model_path", help="مسار النموذج")

    # train
    p_train = sub.add_parser("train", help="تدريب QLoRA")
    p_train.add_argument("--model", type=str, help="مسار النموذج الأساسي")
    p_train.add_argument("--data", type=str, help="ملف البيانات JSONL")
    p_train.add_argument("--output", type=str, help="مجلد المخرجات")
    p_train.add_argument("--lora-r", type=int, default=16)
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--max-seq-len", type=int, default=2048)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--grad-accum", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--warmup", type=int, default=100)
    p_train.add_argument("--save-every", type=int, default=100)
    p_train.add_argument("--no-packing", action="store_true")
    p_train.add_argument("--resume", type=str, help="Checkpoint to resume from")

    # test
    p_test = sub.add_parser("test", help="اختبار النموذج")
    p_test.add_argument("model_path", help="مسار النموذج")
    p_test.add_argument("--prompt", type=str, help="سؤال واحد")
    p_test.add_argument("--max-seq-len", type=int, default=2048)

    # export
    p_export = sub.add_parser("export", help="تصدير GGUF + Ollama")
    p_export.add_argument("model_path", help="مسار النموذج")
    p_export.add_argument("--quant", type=str, default="q4_k_m",
                         choices=["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "q3_k_m"])
    p_export.add_argument("--output", type=str, help="مجلد المخرجات")
    p_export.add_argument("--to-ollama", action="store_true", help="استيراد تلقائي لـ Ollama")
    p_export.add_argument("--ollama-name", type=str, help="اسم النموذج في Ollama")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "check": cmd_check,
        "list": cmd_list,
        "inspect": cmd_inspect,
        "backup": cmd_backup,
        "train": cmd_train,
        "test": cmd_test,
        "export": cmd_export,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
