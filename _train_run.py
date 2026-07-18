#!/usr/bin/env python3
"""Auto-generated training script by LocalTrainer"""
import json, os, sys, time
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("  LocalTrainer - بدء التدريب")
print(f"  2026-07-18 14:02:39")
print("=" * 60)

# GPU temp check
def check_temp():
    try:
        out = os.popen("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits").read().strip()
        return int(out.split("\n")[0])
    except: return 0

temp = check_temp()
if temp > 70:
    print(f"[WARN] GPU temp {temp}C - waiting for cooldown...")
    import time
    while check_temp() > 65:
        time.sleep(30)
    print(f"[OK] Temp now {check_temp()}C")

print("\nLoading model with Unsloth...")
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/sam2/Downloads/data/models/shared/base/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

print("\nLoading dataset...")
records = []
with open("/home/sam2/Downloads/data/all_categories_cleaned.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            try: records.append(json.loads(line))
            except: pass
print(f"  {len(records)} records loaded")

temp_file = "temp_train_data.jsonl"
with open(temp_file, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

dataset = load_dataset("json", data_files=temp_file, split="train")

def formatting_func(examples):
    col = "messages" if "messages" in examples else "conversations"
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in examples[col]]}

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
            ckpt_dir = self.output_dir / f"checkpoint-{state.global_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(ckpt_dir))
            self.tokenizer.save_pretrained(str(ckpt_dir))
            self.last_save = state.global_step
            elapsed = time.time() - self.start_time
            speed = state.global_step / elapsed if elapsed > 0 else 0
            eta = (args.max_steps - state.global_step) / speed if speed > 0 else 0
            temp = check_temp()
            print(f"\n  [CHECKPOINT] step={state.global_step} loss={kwargs.get('logs',{}).get('loss','?')} temp={temp}C ETA={int(eta//60)}m{int(eta%60)}s")

training_args = TrainingArguments(
    output_dir="/home/sam2/Downloads/LocalTrainer/checkpoints/run_20260718_140150",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=0.0002,
    fp16=True,
    bf16=False,
    logging_steps=5,
    save_strategy="no",
    optim="adamw_8bit",
    warmup_steps=100,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=1.0,
    report_to="none",
    dataloader_num_workers=0,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

ckpt_callback = ManualCheckpoint(50, "/home/sam2/Downloads/LocalTrainer/checkpoints/run_20260718_140150", model, tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,
    args=training_args,
    callbacks=[ckpt_callback],
)

resume_from = None
resume_dir = Path("/home/sam2/Downloads/LocalTrainer/checkpoints/run_20260718_140150")
if resume_dir.exists():
    for d in sorted(resume_dir.iterdir()):
        if d.is_dir() and d.name.startswith("checkpoint-"):
            safetensors = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
            if safetensors:
                resume_from = str(d)

if resume_from:
    print(f"\n  Resuming from: {resume_from}")

print("\n" + "=" * 60)
print("  التدريب يبدأ الآن!")
print("=" * 60)

start = datetime.now()
try:
    trainer.train(resume_from_checkpoint=resume_from)

    print("\n  Saving LoRA adapter...")
    lora_dir = "/home/sam2/Downloads/LocalTrainer/checkpoints/run_20260718_140150/final_lora"
    Path(lora_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    print("  Merging LoRA into base model...")
    merged_dir = "/home/sam2/Downloads/LocalTrainer/checkpoints/run_20260718_140150/merged"
    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.merge_and_unload()
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    elapsed = datetime.now() - start
    print(f"\n  Training complete! Time: {elapsed}")
    print(f"  LoRA adapter: {lora_dir}")
    print(f"  Merged model: {merged_dir}")

except Exception as e:
    print(f"\n  [ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)
