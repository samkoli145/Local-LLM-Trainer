#!/usr/bin/env python3
"""
AC-LoRA MVP - Adaptive Curriculum LoRA
======================================
Sensitivity Analysis + Dynamic Rank + Curriculum Learning
+ Online Censoring + Domino Ordering

Works on GTX 1080 Ti (11GB VRAM)

Usage:
  python ac_lora.py train --model <path> --data <path> --epochs 3
  python ac_lora.py test <model_path>
"""

import argparse
import collections
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Environment optimization for multi-core CPU + CUDA ──
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers - دوال مساعدة
# ═══════════════════════════════════════════════════════════════════════════

def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0

def get_vram_total_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0

def get_ram_gb():
    return psutil.virtual_memory().used / 1e9

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
#  Disk Cache - تخزين مؤقت على القرص
# ═══════════════════════════════════════════════════════════════════════════

DISK_CACHE_DIR = Path(__file__).parent / "_cache"


def cache_exists(key: str) -> bool:
    return (DISK_CACHE_DIR / f"{key}.jsonl").exists()


def cache_save(key: str, texts: List[str]):
    DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = DISK_CACHE_DIR / f"{key}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def cache_load(key: str) -> List[str]:
    path = DISK_CACHE_DIR / f"{key}.jsonl"
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    texts.append(json.loads(line)["text"])
                except:
                    pass
    return texts


def cache_cleanup():
    if DISK_CACHE_DIR.exists():
        for f in DISK_CACHE_DIR.glob("*.jsonl"):
            f.unlink()


# ═══════════════════════════════════════════════════════════════════════════
#  Online Censoring - الرقابة عبر الإنترنت
# ═══════════════════════════════════════════════════════════════════════════

class OnlineCensoring:
    """
    filitering العينات حسب z-score.
    يتجاهل العينات السهلة جداً (loss منخفض) والصعبة جداً (outliers).
    النتيجة: تدريب على عينات مفيدة فقط = تقارب أسرع.
    """

    def __init__(self, warmup_steps: int = 50, window_size: int = 100,
                 z_min: float = 0.3, z_max: float = 2.5):
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.z_min = z_min
        self.z_max = z_max
        self.loss_history = collections.deque(maxlen=window_size)
        self.global_step = 0
        self.skipped = 0
        self.kept = 0

    def should_keep(self, loss_value: float) -> bool:
        """هل نحتفظ بهذه العينة؟"""
        self.global_step += 1

        # Warmup: نحتفظ بكل شيء في البداية
        if self.global_step < self.warmup_steps:
            self.kept += 1
            return True

        self.loss_history.append(loss_value)

        if len(self.loss_history) < 20:
            self.kept += 1
            return True

        avg = np.mean(self.loss_history)
        std = np.std(self.loss_history) + 1e-8
        z = abs(loss_value - avg) / std

        if self.z_min < z < self.z_max:
            self.kept += 1
            return True
        else:
            self.skipped += 1
            return False

    def get_stats(self) -> Dict:
        total = self.kept + self.skipped
        return {
            "kept": self.kept,
            "skipped": self.skipped,
            "skip_rate": self.skipped / max(total, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Domino Ordering - ترتيب الدومينو
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Skill:
    name: str
    priority: int = 0
    progress: float = 0.0
    is_learned: bool = False
    samples_seen: int = 0
    total_loss: float = 0.0


class DominoScheduler:
    """
    جدولة المهارات: كل مرحلة صعوبة = مهارة.
    عندما تتعلم المهارة الحالية (loss < threshold)،نتقل للتالية.
    """

    def __init__(self, difficulty_bins: List[str] = None):
        if difficulty_bins is None:
            difficulty_bins = [
                "easy (tokens < 128)",
                "medium (128-512)",
                "hard (512-1024)",
                "expert (1024+)",
            ]
        self.skills = [Skill(name, i) for i, name in enumerate(difficulty_bins)]
        self.current_idx = 0
        self.loss_threshold = 2.0  # avg loss to "learn" a skill

    def get_active(self) -> Optional[Skill]:
        if self.current_idx < len(self.skills):
            return self.skills[self.current_idx]
        return None

    def update(self, avg_loss: float):
        """تحديث تقدم المهارة النشطة"""
        active = self.get_active()
        if active is None:
            return

        active.samples_seen += 1
        active.total_loss += avg_loss
        avg = active.total_loss / active.samples_seen

        # المهارة تمت تعلمها إذا كان avg loss < threshold
        if avg < self.loss_threshold and active.samples_seen >= 20:
            active.is_learned = True
            active.progress = 1.0
            if self.current_idx < len(self.skills) - 1:
                self.current_idx += 1
                next_skill = self.skills[self.current_idx]
                print(f"\n  [DOMINO] Learned '{active.name}' -> moving to '{next_skill.name}'")

    def get_status(self) -> Dict:
        learned = sum(1 for s in self.skills if s.is_learned)
        active = self.get_active()
        return {
            "learned": learned,
            "total": len(self.skills),
            "current": active.name if active else "done",
            "progress": f"{learned}/{len(self.skills)}",
        }


# ═══════════════════════════════════════════════════════════════════════════
#  1. Sensitivity Analysis - تحليل حساسية الطبقات
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LayerSensitivity:
    name: str
    gradient_norm: float
    fisher_info: float
    weight_norm: float
    sensitivity_rank: int  # 0 = most sensitive


class SensitivityAnalyzer:
    """
    يحلل حساسية كل طبقة LoRA باستخدام بيانات حقيقية.
    النتيجة: ترتيب الطبقات حسب أهميتها للتدريب.
    """

    def __init__(self, model, tokenizer, data_file: str,
                 num_samples: int = 32, max_seq_len: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.results: Dict[str, LayerSensitivity] = {}

    def analyze(self) -> Dict[str, LayerSensitivity]:
        """تحليل حساسية كل طبقة LoRA"""
        print("  [1/3] تحميل عينات للتحليل...")
        samples = self._load_samples()

        print("  [2/3] جمع التدرجات...")
        gradients = self._collect_gradients(samples)

        print("  [3/3] حساب الحساسية...")
        self._compute_sensitivity(gradients)

        return self.results

    def _load_samples(self) -> List[dict]:
        """تحميل عينات حقيقية من البيانات"""
        samples = []
        with open(self.data_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.num_samples:
                    break
                if line.strip():
                    try:
                        samples.append(json.loads(line))
                    except:
                        pass
        return samples

    def _collect_gradients(self, samples: List[dict]) -> Dict[str, List[float]]:
        """جمع التدرجات من forward/backward على عينات حقيقية"""
        self.model.train()

        # hooks لجمع التدرجات
        gradients: Dict[str, List[float]] = {}
        hooks = []

        for name, module in self.model.named_modules():
            if not hasattr(module, "weight"):
                continue
            if not module.weight.requires_grad:
                continue
            # فقط الطبقات اللي ممكن نضيف LoRA لها
            if not any(x in name for x in ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"]):
                continue

            def hook_fn(module, grad_input, grad_output, n=name):
                if grad_output is None:
                    return
                grad = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                if grad is not None and grad.numel() > 0:
                    if n not in gradients:
                        gradients[n] = []
                    gradients[n].append(grad.detach().abs().mean().item())

            hook = module.register_full_backward_hook(hook_fn)
            hooks.append(hook)

        # Forward/backward على عينات حقيقية
        for sample in samples:
            try:
                # بناء ChatML prompt
                messages = sample.get("messages", sample.get("conversations", []))
                if not messages:
                    continue
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=self.max_seq_len, padding=False
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # Forward
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # Backward
                loss.backward(retain_graph=True)
                self.model.zero_grad()
            except Exception as e:
                continue

        # إزالة hooks
        for hook in hooks:
            hook.remove()

        self.model.zero_grad()
        return gradients

    def _compute_sensitivity(self, gradients: Dict[str, List[float]]):
        """حساب درجة الحساسية لكل طبقة"""
        # حساب متوسط حجم التدرج لكل طبقة
        avg_grads = {}
        for name, grads in gradients.items():
            if grads:
                avg_grads[name] = np.mean(grads)
            else:
                avg_grads[name] = 0.0

        # حساب Fisher Information = mean(grad^2)
        fisher = {}
        for name, grads in gradients.items():
            if grads:
                fisher[name] = np.mean([g**2 for g in grads])
            else:
                fisher[name] = 0.0

        # حساب حجم الأوزان
        weight_norms = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and module.weight.requires_grad:
                if name in avg_grads:
                    weight_norms[name] = module.weight.detach().norm().item()

        # ترتيب حسب الحساسية (الأعلى = الأكثر حساسية)
        sorted_layers = sorted(avg_grads.items(), key=lambda x: x[1], reverse=True)

        for rank, (name, grad_norm) in enumerate(sorted_layers):
            self.results[name] = LayerSensitivity(
                name=name,
                gradient_norm=grad_norm,
                fisher_info=fisher.get(name, 0.0),
                weight_norm=weight_norms.get(name, 0.0),
                sensitivity_rank=rank,
            )

        # طباعة ملخص
        print(f"\n  {'الطبقة':40s} {'.grad_norm':>12s} {'fisher':>12s} {'rank':>6s}")
        print(f"  {'-'*72}")
        for name, r in list(self.results.items())[:8]:
            print(f"  {name:40s} {r.gradient_norm:12.6f} {r.fisher_info:12.6f} {r.sensitivity_rank:6d}")
        if len(self.results) > 8:
            print(f"  ... +{len(self.results)-8} طبقات أخرى")


# ═══════════════════════════════════════════════════════════════════════════
#  2. Dynamic Rank Allocation - توزيع الرتب الديناميكي
# ═══════════════════════════════════════════════════════════════════════════

def compute_layer_ranks(
    sensitivity: Dict[str, LayerSensitivity],
    base_rank: int = 16,
    min_rank: int = 4,
    max_rank: int = 32,
) -> Dict[str, int]:
    """
    توزيع رتب LoRA لكل طبقة بناءً على الحساسية.
    الطبقات الحساسة = رتبة أعلى (قدرة أكبر).
    الطبقات غير الحساسة = رتبة أقل (وفّر VRAM).
    """
    if not sensitivity:
        return {}

    total = len(sensitivity)
    ranks = {}

    for name, layer in sensitivity.items():
        # نسبة الحساسية: 0 (least) → 1 (most)
        ratio = (total - layer.sensitivity_rank) / max(total - 1, 1)

        # رتبة مقترحة: min_rank → max_rank حسب ratio
        rank = int(min_rank + (max_rank - min_rank) * ratio)
        rank = max(min_rank, min(max_rank, rank))

        # تأكد من أن الرتبة çiftية (أفضل للـ GPU)
        rank = (rank // 2) * 2
        rank = max(min_rank, rank)

        ranks[name] = rank

    return ranks


# ═══════════════════════════════════════════════════════════════════════════
#  3. Curriculum Learning - تعلم منهجي
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumStage:
    name: str
    difficulty_threshold: float  # 0.0 - 1.0
    batch_size: int
    lr_multiplier: float


class CurriculumScheduler:
    """
    ترتيب البيانات من السهل إلى الصعب تدريجياً.
    الصعوبة = طول التسلسل + تعقيد المحتوى.
    """

    def __init__(self, total_epochs: int = 3):
        self.total_epochs = total_epochs
        self.stages = self._create_stages()

    def _create_stages(self) -> List[CurriculumStage]:
        if self.total_epochs <= 1:
            return [CurriculumStage("full", 1.0, 2, 1.0)]

        if self.total_epochs == 2:
            return [
                CurriculumStage("easy", 0.5, 4, 0.5),
                CurriculumStage("full", 1.0, 2, 1.0),
            ]

        return [
            CurriculumStage("easy", 0.33, 8, 0.3),
            CurriculumStage("medium", 0.66, 4, 0.7),
            CurriculumStage("hard", 1.0, 2, 1.0),
        ]

    def get_stage(self, epoch: int) -> CurriculumStage:
        idx = min(epoch, len(self.stages) - 1)
        return self.stages[idx]


def compute_difficulty(texts: List[str], tokenizer) -> List[float]:
    """
    حساب صعوبة كل نص.
    الصعوبة = طول التسلسل + تنوع المفردات + تعقيد الكلمات.
    """
    # كلمات معقدة (إنجليزية شائعة في النصوص التقنية)
    complex_words = {
        'however', 'therefore', 'consequently', 'significant',
        'implementation', 'infrastructure', 'optimization',
        'configuration', 'environment', 'performance', 'architecture',
        'simultaneously', 'comprehensive', 'fundamental', 'methodology',
    }

    difficulties = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)

        # 1. طول التسلسل (0.5 وزن)
        length_score = min(token_count / 2048.0, 1.0)

        # 2. تنوع المفردات (0.3 وزن)
        words = text.split()
        word_count = len(words)
        unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
        vocab_score = unique_ratio * 0.5

        # 3. كلمات معقدة (0.2 وزن)
        complex_count = sum(1 for w in words if w.lower() in complex_words)
        complex_score = min(complex_count / max(word_count, 1) * 5, 1.0)

        difficulty = length_score * 0.5 + vocab_score * 0.3 + complex_score * 0.2
        difficulties.append(min(1.0, difficulty))

    return difficulties


def sort_by_curriculum(
    data: List[dict], tokenizer, epoch: int, total_epochs: int
) -> List[dict]:
    """
    ترتيب البيانات حسب المنهج التعليمي.
    الحقبة الأولى: أقصر 33% فقط.
    الحقبة الثانية: أقصر 66%.
    الحقبة الثالثة: كل البيانات.
    """
    if total_epochs <= 1:
        return data

    # حساب الصعوبة لكل عنصر
    texts = []
    for item in data:
        messages = item.get("messages", item.get("conversations", []))
        if messages:
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ))
        else:
            texts.append("")

    difficulties = compute_difficulty(texts, tokenizer)

    # دمج البيانات مع الصعوبة
    paired = list(zip(data, difficulties))
    paired.sort(key=lambda x: x[1])  # ترتيب من الأسهل إلى الأصعب

    # تحديد النسبة حسب الحقبة
    ratio = min(1.0, (epoch + 1) / total_epochs)
    cutoff = int(len(paired) * ratio)
    cutoff = max(cutoff, 16)  # على الأقل 16 عنصر

    selected = [item for item, diff in paired[:cutoff]]

    print(f"  Curriculum: epoch {epoch+1}/{total_epochs} "
          f"- ratio={ratio:.2f} - {len(selected)}/{len(data)} examples")

    return selected


# ═══════════════════════════════════════════════════════════════════════════
#  4. AC-LoRA Trainer - محرك التدريب
# ═══════════════════════════════════════════════════════════════════════════

def train_ac_lora(args):
    """التدريب مع AC-LoRA"""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback
    from datasets import Dataset

    print("=" * 60)
    print("  AC-LoRA MVP - Adaptive Curriculum LoRA")
    print("=" * 60)

    # ── GPU check ──
    if not torch.cuda.is_available():
        print("  [ERR] No GPU!")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
    bf16 = cc >= 70
    print(f"  GPU: {gpu_name} ({vram_gb:.1f}GB) BF16={bf16}")

    # ── تحميل النموذج ──
    print(f"\n  تحميل النموذج: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        use_cache=False,  # توفير VRAM للـ KV cache
    )
    cleanup_memory()
    print(f"  VRAM after load: {get_vram_gb():.2f} GB")

    # ── Sensitivity Analysis ──
    print(f"\n  ── Sensitivity Analysis ──")
    analyzer = SensitivityAnalyzer(
        model=model,
        tokenizer=tokenizer,
        data_file=args.data,
        num_samples=args.sensitivity_samples,
        max_seq_len=min(args.max_seq_len, 512),
    )
    sensitivity = analyzer.analyze()

    # ── Dynamic Rank Allocation ──
    print(f"\n  ── Dynamic Rank Allocation ──")
    layer_ranks = compute_layer_ranks(
        sensitivity,
        base_rank=args.lora_r,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )

    # تجميع الرتب الفريدة
    unique_ranks = sorted(set(layer_ranks.values()))
    total_params = sum(r for r in layer_ranks.values())
    avg_rank = np.mean(list(layer_ranks.values()))
    print(f"  Unique ranks: {unique_ranks}")
    print(f"  Avg rank: {avg_rank:.1f}")
    print(f"  Total LoRA params budget: {total_params}")

    # ── تطبيق LoRA مع رتب ديناميكية ──
    print(f"\n  تطبيق LoRA مع Dynamic Rank...")

    # بناء target_modules مع rank لكل طبقة
    # PEFT لا يدعم rank مختلف لكل طبقة مباشرة
    # الحل: نستخدم rank متوسط ونعدّل يدوياً بعد التطبيق
    target_modules = list(set(
        name.split(".")[-1]
        for name in layer_ranks.keys()
    ))

    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=args.lora_r,  # سنعدّل لكل طبقة بعد التطبيق
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_dora=True,  # DoRA: Weight-Decomposed Low-Rank Adaptation
    )

    model = get_peft_model(model, lora_config)

    # تعديل الرتب لكل طبقة
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        # البحث عن الاسم المطابق في layer_ranks
        for layer_name, target_rank in layer_ranks.items():
            if layer_name.split(".")[-1] in name and "lora" in name:
                if hasattr(module, "r"):
                    old_r = module.r
                    module.r = target_rank
                    module.lora_A = nn.Parameter(
                        module.lora_A.data[:target_rank, :]
                    )
                    module.lora_B = nn.Parameter(
                        module.lora_B.data[:, :target_rank]
                    )
                    break

    # طباعة إحصائيات النموذج
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    cleanup_memory()
    print(f"  VRAM after LoRA: {get_vram_gb():.2f} GB")

    # ── تحميل البيانات ──
    print(f"\n  تحميل البيانات: {args.data}")
    all_data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except:
                    pass
    print(f"  {len(all_data)} records loaded")

    # ── Online Censoring + Domino ──
    censoring = OnlineCensoring(
        warmup_steps=50, window_size=100, z_min=0.3, z_max=2.5
    )
    domino = DominoScheduler(difficulty_bins=[
        "easy (tokens < 128)",
        "medium (128-512)",
        "hard (512-1024)",
        "expert (1024+)",
    ])
    print(f"  Censoring: warmup=50, z=[0.3, 2.5]")
    print(f"  Domino: {len(domino.skills)} skills")

    # ── تحميل البيانات ──
    print(f"\n  تحميل البيانات: {args.data}")
    all_data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except:
                    pass
    print(f"  {len(all_data)} records loaded")

    # ── تجهيز البيانات ──
    print(f"\n  ── Data Preparation ──")
    curriculum = CurriculumScheduler(total_epochs=args.epochs)

    # Check disk cache
    cache_key = f"sorted_{Path(args.data).stem}_{len(all_data)}"
    if cache_exists(cache_key):
        print(f"  Loading from disk cache: {cache_key}")
        sorted_texts = cache_load(cache_key)
        paired = [( {"text": t}, compute_difficulty([t], tokenizer)[0] ) for t in sorted_texts]
    else:
        def format_example(example):
            messages = example.get("messages", example.get("conversations", []))
            if not messages:
                return {"text": ""}
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        formatted = [format_example(item) for item in all_data]
        formatted = [f for f in formatted if f["text"]]

        # Compute difficulty and sort (Domino ordering)
        difficulties = compute_difficulty([f["text"] for f in formatted], tokenizer)
        paired = list(zip(formatted, difficulties))
        paired.sort(key=lambda x: x[1])

        # Save to disk cache
        cache_save(cache_key, [f["text"] for f, _ in paired])
        print(f"  Saved to disk cache: {cache_key}")

    # Censoring: remove outlier samples by difficulty
    if len(paired) > 100:
        diff_arr = np.array([d for _, d in paired])
        diff_mean = np.mean(diff_arr)
        diff_std = np.std(diff_arr) + 1e-8
        z_scores = np.abs(diff_arr - diff_mean) / diff_std
        # Keep samples with z < 3 (remove extreme outliers)
        mask = z_scores < 3.0
        before = len(paired)
        paired = [p for p, keep in zip(paired, mask) if keep]
        print(f"  Censoring: removed {before - len(paired)} outlier samples ({len(paired)} remaining)")

    # Assign to domino bins
    for text, diff in paired:
        tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if tokens < 128:
            bin_idx = 0
        elif tokens < 512:
            bin_idx = 1
        elif tokens < 1024:
            bin_idx = 2
        else:
            bin_idx = 3
        if bin_idx < len(domino.skills):
            domino.skills[bin_idx].samples_seen += 1

    print(f"  Domino bins: {[(s.name, s.samples_seen) for s in domino.skills]}")

    # ── Checkpoint callback ──
    output_dir = args.output or str(
        Path(__file__).parent / "checkpoints" / f"ac_lora_{int(time.time())}"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    class ManualCheckpoint(TrainerCallback):
        def __init__(self, save_every, output_dir, model, tokenizer,
                     censoring=None, domino=None):
            self.save_every = save_every
            self.output_dir = Path(output_dir)
            self.model = model
            self.tokenizer = tokenizer
            self.last_save = 0
            self.start_time = time.time()
            self.censoring = censoring
            self.domino = domino

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
                loss = kwargs.get("logs", {}).get("loss", "?")
                extra = ""
                if self.censoring:
                    cs = self.censoring.get_stats()
                    extra += f" skip={cs['skip_rate']:.1%}"
                if self.domino:
                    ds = self.domino.get_status()
                    extra += f" domino={ds['progress']}"
                print(f"\n  [CKPT] step={state.global_step} loss={loss} "
                      f"ETA={int(eta//60)}m{int(eta%60)}s{extra}")

    # ── بناء مجموعات البيانات لكل حقبة ──
    def make_dataset(texts):
        temp = Path(output_dir) / f"temp_{len(texts)}.jsonl"
        with open(temp, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        ds = Dataset.from_json(str(temp))
        temp.unlink(missing_ok=True)
        return ds

    effective_batch = args.batch_size * args.grad_accum

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        learning_rate=args.lr,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=5,
        save_strategy="no",
        optim="paged_adamw_8bit",
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    ckpt_callback = ManualCheckpoint(
        args.save_every, output_dir, model, tokenizer,
        censoring=censoring, domino=domino,
    )

    print(f"\n{'='*60}")
    print(f"  AC-LoRA Training Starts!")
    print(f"  Dynamic ranks: {unique_ranks}")
    print(f"  Curriculum: easy→hard (Domino + Censoring)")
    print(f"{'='*60}\n")

    start = time.time()
    batch_size = args.batch_size
    try:
        # Epoch 1: Easy (50% easiest)
        easy_texts = [item["text"] for item, _ in paired[:int(len(paired) * 0.5)]]
        dataset_1 = make_dataset(easy_texts)
        steps_1 = len(dataset_1) // effective_batch
        print(f"\n  === Epoch 1/{args.epochs}: EASY ({len(easy_texts)} samples, {steps_1} steps) ===")
        print(f"  VRAM: {get_vram_gb():.2f}/{get_vram_total_gb():.1f} GB | RAM: {get_ram_gb():.1f} GB")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_1,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=True,
            args=training_args,
            callbacks=[ckpt_callback],
        )
        trainer.train()

        # Epoch 2: Medium (75% easiest)
        if args.epochs >= 2:
            med_texts = [item["text"] for item, _ in paired[:int(len(paired) * 0.75)]]
            dataset_2 = make_dataset(med_texts)
            steps_2 = len(dataset_2) // effective_batch
            print(f"\n  === Epoch 2/{args.epochs}: MEDIUM ({len(med_texts)} samples, {steps_2} steps) ===")
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset_2,
                dataset_text_field="text",
                max_seq_length=args.max_seq_len,
                packing=True,
                args=training_args,
                callbacks=[ckpt_callback],
            )
            trainer.train()

        # Epoch 3: Hard (all data)
        if args.epochs >= 3:
            all_texts = [item["text"] for item, _ in paired]
            dataset_3 = make_dataset(all_texts)
            steps_3 = len(dataset_3) // effective_batch
            print(f"\n  === Epoch 3/{args.epochs}: HARD ({len(all_texts)} samples, {steps_3} steps) ===")
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset_3,
                dataset_text_field="text",
                max_seq_length=args.max_seq_len,
                packing=True,
                args=training_args,
                callbacks=[ckpt_callback],
            )
            trainer.train()

        # Save final
        lora_dir = str(Path(output_dir) / "final_lora")
        Path(lora_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)

        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"  AC-LoRA Training Complete!")
        print(f"  Time: {elapsed/60:.1f} minutes")
        print(f"  LoRA adapter: {lora_dir}")
        print(f"  Dynamic ranks used: {unique_ranks}")
        cs = censoring.get_stats()
        ds = domino.get_status()
        print(f"  Censoring: kept={cs['kept']} skipped={cs['skipped']} ({cs['skip_rate']:.1%} skip rate)")
        print(f"  Domino: {ds['learned']}/{ds['total']} skills learned")
        print(f"{'='*60}")

    except KeyboardInterrupt:
        print("\n  [STOP] Interrupted. Checkpoint saved if any.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n  [OOM] VRAM exhausted! Current: {get_vram_gb():.2f} GB")
            print(f"  [OOM] Try: --batch-size {max(1, batch_size-1)} --max-seq-len 1024")
            cleanup_memory()
        else:
            raise e
    except Exception as e:
        print(f"\n  [ERR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp files
        for f in Path(output_dir).glob("temp_train_*.jsonl"):
            f.unlink(missing_ok=True)
        cleanup_memory()


# ═══════════════════════════════════════════════════════════════════════════
#  Interactive Test - اختبار تفاعلي
# ═══════════════════════════════════════════════════════════════════════════

def test_ac_lora(args):
    """اختبار النموذج المدرب تفاعلياً"""
    from unsloth import FastLanguageModel

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"  [ERR] Model not found: {model_path}")
        return

    print(f"\n  تحميل النموذج: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        use_cache=False,
    )
    FastLanguageModel.for_inference(model)
    cleanup_memory()

    print(f"  VRAM: {get_vram_gb():.2f} GB")
    print(f"\n  اكتب سؤالاً (اكتب 'quit' للخروج):")
    print(f"  {'─'*50}")

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
        print(f"  ({tokens} tokens in {elapsed:.1f}s = {tokens/elapsed:.1f} tok/s)")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AC-LoRA: Adaptive Curriculum LoRA Training"
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # train
    p_train = sub.add_parser("train", help="Train with AC-LoRA")
    p_train.add_argument("--model", type=str, required=True,
                        help="Base model path")
    p_train.add_argument("--data", type=str, required=True,
                        help="Training data JSONL")
    p_train.add_argument("--output", type=str, default="",
                        help="Output directory")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--grad-accum", type=int, default=4)
    p_train.add_argument("--warmup", type=int, default=100)
    p_train.add_argument("--max-seq-len", type=int, default=2048)
    p_train.add_argument("--save-every", type=int, default=100)

    # AC-LoRA specific
    p_train.add_argument("--lora-r", type=int, default=16,
                        help="Base LoRA rank")
    p_train.add_argument("--lora-alpha", type=int, default=32)
    p_train.add_argument("--min-rank", type=int, default=4,
                        help="Minimum LoRA rank for insensitive layers")
    p_train.add_argument("--max-rank", type=int, default=32,
                        help="Maximum LoRA rank for sensitive layers")
    p_train.add_argument("--sensitivity-samples", type=int, default=32,
                        help="Number of samples for sensitivity analysis")

    # test
    p_test = sub.add_parser("test", help="Interactive test")
    p_test.add_argument("model_path", help="Path to trained model/adapter")
    p_test.add_argument("--max-seq-len", type=int, default=2048)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "train":
        train_ac_lora(args)
    elif args.command == "test":
        test_ac_lora(args)


if __name__ == "__main__":
    main()
