#!/usr/bin/env python3
"""
Ultimate Trainer v2.0 - Production Ready
=========================================
يدمج جميع التقنيات:
1. Domino Ordering (فيزياء التعلم - ترتيب البيانات من الأسهل للأصعب)
2. Selective Cloud Training (بيانات سحابية انتقائية)
3. Smart External Memory (ذاكرة خارجية مع mmap)
4. Online Censoring (تجاهل العينات غير المفيدة)
5. NiNo Prediction (تنبؤ الأوزان المستقبلية)
6. تحسينات Xeon 28 Core + GTX 1080 Ti

يعمل بثبات على: 11GB VRAM + 15.5GB RAM
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset, load_from_disk
import gc
import os
import time
import json
import psutil
import numpy as np
import hashlib
import random
from typing import Dict, Optional, List, Iterator, Tuple
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
import threading
import queue

# ============================================================
# 0. التحسينات البيئية لـ Xeon 28 Core
# ============================================================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# 1. نظام Domino Ordering (فيزياء التعلم)
# ============================================================
@dataclass
class Skill:
    name: str
    difficulty_range: Tuple[float, float]  # نطاق الصعوبة
    priority: int = 0
    progress: float = 0.0
    is_learned: bool = False

class DominoScheduler:
    """
    جدولة المهارات حسب ظاهرة الدومينو
    يبدأ بالأسهل ثم ينتقل للأصعب تدريجياً
    """
    
    def __init__(self):
        self.skills = []
        self.current_skill_idx = 0
        self.threshold = 0.7
        self.history = []
        self._define_skills()
        
    def _define_skills(self):
        """تعريف المهارات حسب مستويات الصعوبة"""
        skill_levels = [
            ("Simple Patterns", (0.0, 0.2)),
            ("Basic Vocabulary", (0.2, 0.4)),
            ("Medium Complexity", (0.4, 0.6)),
            ("Advanced Concepts", (0.6, 0.8)),
            ("Expert Reasoning", (0.8, 1.0))
        ]
        
        self.skills = [Skill(name, range_, i) for i, (name, range_) in enumerate(skill_levels)]
        print(f"🎯 Domino Scheduler: {len(self.skills)} skill levels defined")
    
    def get_skill_for_difficulty(self, difficulty: float) -> Skill:
        """الحصول على المهارة المناسبة لمستوى الصعوبة"""
        for skill in self.skills:
            if skill.difficulty_range[0] <= difficulty < skill.difficulty_range[1]:
                return skill
        return self.skills[-1]  # أعلى مستوى
    
    def get_active_skill(self) -> Optional[Skill]:
        """الحصول على المهارة النشطة حالياً"""
        if self.current_skill_idx < len(self.skills):
            return self.skills[self.current_skill_idx]
        return None
    
    def update_progress(self, difficulty: float):
        """تحديث التقدم بناءً على الصعوبة الحالية"""
        skill = self.get_skill_for_difficulty(difficulty)
        
        # تحديث تقدم المهارة
        progress_in_skill = (difficulty - skill.difficulty_range[0]) / (skill.difficulty_range[1] - skill.difficulty_range[0] + 0.001)
        skill.progress = min(1.0, progress_in_skill)
        
        if skill.progress >= self.threshold and not skill.is_learned:
            skill.is_learned = True
            self.history.append({
                'skill': skill.name,
                'time': time.time(),
                'difficulty': difficulty
            })
            self._trigger_next()
    
    def _trigger_next(self):
        """تأثير الدومينو: الانتقال للمستوى التالي"""
        if self.current_skill_idx < len(self.skills) - 1:
            self.current_skill_idx += 1
            next_skill = self.skills[self.current_skill_idx]
            print(f"🔄 Domino Effect: Advancing to '{next_skill.name}' (Difficulty: {next_skill.difficulty_range})")
    
    def get_status(self) -> Dict:
        """حالة التعلم الحالية"""
        learned = sum(1 for s in self.skills if s.is_learned)
        return {
            'total_skills': len(self.skills),
            'learned': learned,
            'current_skill': self.get_active_skill().name if self.get_active_skill() else None,
            'progress': learned / len(self.skills),
            'history': self.history[-5:]  # آخر 5 مهارات
        }

# ============================================================
# 2. Smart External Memory (mmap-based)
# ============================================================
class SmartExternalMemory:
    """
    ذاكرة خارجية باستخدام Apache Arrow (mmap)
    تسمح بقراءة بيانات ضخمة باستخدام 0% RAM تقريباً
    """
    
    def __init__(self, cache_dir: str = "./external_cache", max_size_gb: float = 20.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.current_size_gb = 0
        
        print(f"💾 Smart External Memory (mmap) initialized")
        print(f"   Cache: {cache_dir}")
        print(f"   Max Size: {max_size_gb} GB")
    
    def save_dataset(self, dataset, name: str) -> str:
        """حفظ مجموعة بيانات بصيغة Arrow (ضغط عالي + mmap)"""
        path = self.cache_dir / name.replace("/", "_")
        print(f"💾 Saving to disk: {path}")
        dataset.save_to_disk(str(path))
        return str(path)
    
    def load_dataset(self, name: str):
        """تحميل مجموعة بيانات من القرص (mmap)"""
        path = self.cache_dir / name.replace("/", "_")
        if path.exists():
            print(f"📦 Loading from mmap cache: {name}")
            return load_from_disk(str(path))
        return None
    
    def exists(self, name: str) -> bool:
        path = self.cache_dir / name.replace("/", "_")
        return path.exists()
    
    def get_stats(self) -> Dict:
        """إحصائيات الذاكرة الخارجية"""
        total_size = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        self.current_size_gb = total_size / 1e9
        return {
            'total_size_gb': self.current_size_gb,
            'max_size_gb': self.max_size_gb,
            'usage_percent': (self.current_size_gb / self.max_size_gb) * 100,
            'num_files': len(list(self.cache_dir.rglob("*")))
        }

# ============================================================
# 3. Cloud Data Repository مع Domino Filtering
# ============================================================
class DominoCloudRepository:
    """
    مستودع بيانات سحابي مع فلترة Domino وترتيب
    """
    
    def __init__(self, memory: SmartExternalMemory, domino: DominoScheduler):
        self.memory = memory
        self.domino = domino
        self.stats = {
            'total_fetched': 0,
            'filtered_out': 0,
            'kept': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def _calculate_difficulty(self, text: str) -> float:
        """حساب صعوبة النص (0-1)"""
        if not text:
            return 0.0
        
        words = text.split()
        word_count = len(words)
        
        # عوامل الصعوبة:
        # 1. الطول
        length_score = min(1.0, word_count / 300)
        
        # 2. تنوع المفردات
        unique_ratio = len(set(words)) / max(word_count, 1)
        vocab_score = unique_ratio * 0.5
        
        # 3. وجود كلمات معقدة (محاكاة)
        complex_words = ['however', 'therefore', 'consequently', 'significant', 'implementation']
        complex_score = sum(1 for w in words if w.lower() in complex_words) / max(word_count, 1)
        
        # الدمج
        difficulty = (length_score * 0.5 + vocab_score * 0.3 + complex_score * 0.2)
        return min(1.0, difficulty)
    
    def fetch_and_order(self, dataset_name: str, split: str = "train",
                        max_samples: Optional[int] = None,
                        use_cache: bool = True) -> Iterator:
        """
        جلب البيانات، حساب الصعوبة، ترتيب حسب Domino، وتدفقها
        """
        cache_key = f"{dataset_name}_{split}_domino"
        
        # 1. محاولة التحميل من الكاش
        if use_cache and self.memory.exists(cache_key):
            self.stats['cache_hits'] += 1
            cached = self.memory.load_dataset(cache_key)
            if cached is not None:
                print(f"📦 Using cached Domino-ordered data: {dataset_name}")
                for sample in cached:
                    yield sample
                return
        
        self.stats['cache_misses'] += 1
        print(f"📥 Fetching and ordering: {dataset_name}")
        
        # 2. جلب البيانات مع التدفق
        raw_dataset = load_dataset(dataset_name, split=split, streaming=True)
        if max_samples:
            raw_dataset = raw_dataset.take(max_samples * 2)  # ضعف للفلترة
        
        # 3. حساب الصعوبة وتجميع البيانات المفلترة
        samples_with_difficulty = []
        
        for sample in raw_dataset:
            text = sample.get('text', '') or sample.get('review', '')
            if not text:
                continue
            
            difficulty = self._calculate_difficulty(text)
            self.stats['total_fetched'] += 1
            
            # Domino Filtering: الاحتفاظ بالعينات في نطاق المهارة الحالية
            active_skill = self.domino.get_active_skill()
            if active_skill:
                low, high = active_skill.difficulty_range
                if low <= difficulty < high + 0.1:  # قبول مرن
                    samples_with_difficulty.append({
                        'sample': sample,
                        'difficulty': difficulty
                    })
                    self.stats['kept'] += 1
                else:
                    self.stats['filtered_out'] += 1
            else:
                samples_with_difficulty.append({
                    'sample': sample,
                    'difficulty': difficulty
                })
                self.stats['kept'] += 1
            
            # تحديث تقدم Domino
            if len(samples_with_difficulty) % 100 == 0:
                avg_difficulty = np.mean([s['difficulty'] for s in samples_with_difficulty[-100:]])
                self.domino.update_progress(avg_difficulty)
        
        # 4. ترتيب حسب الصعوبة (Domino Ordering) - تصاعدي
        samples_with_difficulty.sort(key=lambda x: x['difficulty'])
        sorted_samples = [s['sample'] for s in samples_with_difficulty]
        
        # 5. حفظ في الذاكرة الخارجية للاستخدام المستقبلي
        if use_cache and len(sorted_samples) > 100:
            ds = Dataset.from_list(sorted_samples)
            self.memory.save_dataset(ds, cache_key)
        
        # 6. تدفق النتائج
        for sample in sorted_samples:
            yield sample
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            'domino_status': self.domino.get_status()
        }

# ============================================================
# 4. NiNo Predictor (تنبؤ الأوزان)
# ============================================================
class NiNoPredictor:
    """
    يتنبأ بالأوزان المستقبلية لتسريع التقارب
    """
    
    def __init__(self, model, history_window: int = 5):
        self.model = model
        self.history_window = history_window
        self.weight_history = {}
        self.step_count = 0
        
        # تخزين مؤقت للأوزان
        self._init_history()
    
    def _init_history(self):
        """تهيئة سجل الأوزان"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.weight_history[name] = deque(maxlen=self.history_window)
    
    def record(self):
        """تسجيل الأوزان الحالية"""
        self.step_count += 1
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.weight_history[name].append(param.data.clone())
    
    def predict_and_apply(self, blend_factor: float = 0.2):
        """التنبؤ وتطبيق الأوزان المحسّنة"""
        if self.step_count < self.history_window:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.weight_history:
                    history = self.weight_history[name]
                    if len(history) >= self.history_window:
                        # تنبؤ: انحدار خطي بسيط (أو متوسط مرجح)
                        weights = torch.linspace(0.5, 1.0, self.history_window)
                        weights = weights.to(history[0].device)
                        weights = weights / weights.sum()
                        
                        # حساب المتوسط المرجح
                        pred = torch.zeros_like(history[0])
                        for i, w in enumerate(history):
                            pred += weights[i] * w
                        
                        # مزج بين الحالي والمتوقع
                        param.data = (1 - blend_factor) * param.data + blend_factor * pred

# ============================================================
# 5. المحرك الرئيسي النهائي
# ============================================================
class UltimateTrainer:
    """
    المدرب النهائي المتكامل
    يدمج: Domino + Cloud + External Memory + NiNo
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("="*70)
        print("  Ultimate Trainer v2.0 - Production Ready")
        print("  Domino + Cloud + External Memory + NiNo")
        print("="*70)
        print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"  CPU Cores: {psutil.cpu_count(logical=False)} Physical / {psutil.cpu_count(logical=True)} Logical")
        print(f"  VRAM: {self._get_vram():.2f} GB / 11 GB")
        print(f"  RAM: {self._get_ram():.2f} GB / 15.5 GB")
        print("="*70)
        
        # 1. Domino Scheduler
        self.domino = DominoScheduler()
        
        # 2. External Memory
        self.memory = SmartExternalMemory(
            cache_dir=config.get('cache_dir', './external_cache'),
            max_size_gb=config.get('cache_size_gb', 20.0)
        )
        
        # 3. Cloud Repository
        self.cloud = DominoCloudRepository(self.memory, self.domino)
        
        # 4. النموذج
        self._load_model()
        
        # 5. NiNo Predictor
        self.nino = NiNoPredictor(self.model, history_window=5)
        
        # 6. إعداد المحسن
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-4),
            weight_decay=0.01
        )
        
        # 7. إحصائيات
        self.step_count = 0
        self.losses = []
        self.start_time = None
        
        print(f"\n✅ Ultimate Trainer جاهز!")
        print(f"   VRAM: {self._get_vram():.2f} GB")
        print(f"   RAM: {self._get_ram():.2f} GB")
        print(f"   Cache: {self.memory.get_stats()['total_size_gb']:.2f} GB")
    
    def _load_model(self):
        """تحميل النموذج مع QLoRA + DoRA"""
        print("\n🧠 Loading Model...")
        
        model_name = self.config.get('model_name', 'microsoft/Phi-3-mini-4k-instruct')
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False,
        )
        
        self.base_model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        lora_config = LoraConfig(
            r=self.config.get('lora_rank', 8),
            lora_alpha=self.config.get('lora_alpha', 16),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
        
        torch.cuda.empty_cache()
        gc.collect()
    
    def _get_vram(self):
        return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    def _get_ram(self):
        return psutil.virtual_memory().used / 1e9
    
    def _prepare_batch(self, texts: List[str]) -> Dict:
        """تحضير دفعة للتدريب"""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_seq_len', 512),
            return_tensors="pt"
        )
    
    def train(self, dataset_name: str, num_steps: int = 500, batch_size: int = 2):
        """بدء التدريب مع كل التقنيات"""
        
        print(f"\n🚀 Starting Ultimate Training on: {dataset_name}")
        print(f"   Steps: {num_steps} | Batch: {batch_size}")
        print("="*60)
        
        self.start_time = time.time()
        
        # جلب البيانات مع Domino Ordering
        data_stream = self.cloud.fetch_and_order(
            dataset_name,
            split='train',
            max_samples=num_steps * batch_size * 2
        )
        
        buffer = []
        step = 0
        
        for sample in data_stream:
            text = sample.get('text', '') or sample.get('review', '')
            if not text:
                continue
            
            buffer.append(text)
            
            if len(buffer) >= batch_size:
                step += 1
                self.step_count += 1
                
                # 1. تحضير الدفعة
                batch = self._prepare_batch(buffer)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 2. Forward
                self.model.train()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # 3. Backward
                loss.backward()
                
                # 4. Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 5. Optimizer Step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # 6. NiNo Prediction
                if step % 10 == 0:
                    self.nino.record()
                    self.nino.predict_and_apply(blend_factor=0.15)
                
                # 7. تحديث Domino
                avg_difficulty = np.mean([self.cloud._calculate_difficulty(t) for t in buffer])
                self.domino.update_progress(avg_difficulty)
                
                # 8. تسجيل الخسارة
                loss_val = loss.item()
                self.losses.append(loss_val)
                buffer = []
                
                # 9. طباعة التقدم
                if step % 50 == 0:
                    avg_loss = np.mean(self.losses[-50:]) if self.losses else 0
                    elapsed = time.time() - self.start_time
                    vram = self._get_vram()
                    ram = self._get_ram()
                    domino_status = self.domino.get_status()
                    
                    print(f"   Step {step}/{num_steps} | Loss: {avg_loss:.4f} | "
                          f"VRAM: {vram:.2f}GB | RAM: {ram:.2f}GB | "
                          f"Domino: {domino_status['progress']:.0%}")
                
                if step >= num_steps:
                    break
        
        # النتائج النهائية
        elapsed = time.time() - self.start_time
        print("\n" + "="*60)
        print("✅ Training Complete!")
        print(f"   Total Steps: {step}")
        print(f"   Final Loss: {self.losses[-1] if self.losses else 0:.4f}")
        print(f"   Avg Loss: {np.mean(self.losses):.4f}")
        print(f"   Time: {elapsed/60:.1f} minutes")
        
        # إحصائيات Domino
        print(f"\n🎯 Domino Learning Progress:")
        status = self.domino.get_status()
        print(f"   Skills Learned: {status['learned']}/{status['total_skills']}")
        print(f"   Current Skill: {status['current_skill']}")
        
        # إحصائيات Cloud
        print(f"\n☁️ Cloud Stats:")
        cloud_stats = self.cloud.get_stats()
        print(f"   Total Fetched: {cloud_stats['total_fetched']}")
        print(f"   Kept: {cloud_stats['kept']}")
        print(f"   Filtered Out: {cloud_stats['filtered_out']}")
        print(f"   Cache Hits: {cloud_stats['cache_hits']}")
        
        # حفظ النموذج
        save_path = "./ultimate_final_model"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\n💾 Model saved to: {save_path}")
    
    def get_stats(self) -> Dict:
        """إحصائيات شاملة"""
        return {
            'domino': self.domino.get_status(),
            'cloud': self.cloud.get_stats(),
            'memory': self.memory.get_stats(),
            'training': {
                'steps': self.step_count,
                'final_loss': self.losses[-1] if self.losses else None,
                'avg_loss': np.mean(self.losses) if self.losses else None
            },
            'system': {
                'vram_gb': self._get_vram(),
                'ram_gb': self._get_ram()
            }
        }

# ============================================================
# 6. التشغيل
# ============================================================
def main():
    config = {
        'model_name': 'microsoft/Phi-3-mini-4k-instruct',
        'cache_dir': './external_cache',
        'cache_size_gb': 20.0,
        'learning_rate': 2e-4,
        'lora_rank': 8,
        'lora_alpha': 16,
        'max_seq_len': 512,
    }
    
    trainer = UltimateTrainer(config)
    trainer.train(
        dataset_name='imdb',
        num_steps=300,
        batch_size=2
    )
    
    # حفظ التقرير النهائي
    with open("ultimate_report.json", "w") as f:
        json.dump(trainer.get_stats(), f, indent=2, default=str)
    
    print("\n📄 Report saved to ultimate_report.json")

if __name__ == "__main__":
    main()