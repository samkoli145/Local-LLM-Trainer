"""
معالجة واستخلاص البيانات من المحادثات
يستخدم Ollama محلياً للاستخلاص الذكي
"""
import re
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from backend.config import settings
from backend.schemas import (
    ExtractionRequest, ExtractionResult, QAPair,
    ExtractionStyle, ExtractionQuality
)
from backend.ollama_client import ollama_client

class DataProcessor:
    """معالجة البيانات واستخلاص أزواج التدريب"""
    
    # أنماط Regex للتعرف على المحادثات
    PATTERNS = {
        'arabic_user': r'(?:المستخدم|مستخدم|أنا|س)\s*[:：]\s*(.+?)(?=\n|$)',
        'arabic_assistant': r'(?:المساعد|مساعد|بوت|AI)\s*[:：]\s*(.+?)(?=\n|$)',
        'english_user': r'(?:User|Human|Me)\s*[:：]\s*(.+?)(?=\n|$)',
        'english_assistant': r'(?:Assistant|Bot|AI)\s*[:：]\s*(.+?)(?=\n|$)',
    }
    
    async def extract_pairs(self, request: ExtractionRequest) -> ExtractionResult:
        """استخلاص أزواج Q&A من نص محادثة"""
        lines = request.text.split('\n')
        raw_pairs = []
        
        # 1. الاستخراج الأولي بـ Regex
        current_instruction = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # كشف السؤال
            is_question = any(
                re.match(p, line, re.IGNORECASE)
                for p in [self.PATTERNS['arabic_user'], self.PATTERNS['english_user']]
            )
            
            if is_question:
                # استخراج النص بعد الـ prefix
                for pattern in [self.PATTERNS['arabic_user'], self.PATTERNS['english_user']]:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        current_instruction = match.group(1).strip()
                        break
            
            # كشف الإجابة
            elif current_instruction:
                is_answer = any(
                    re.match(p, line, re.IGNORECASE)
                    for p in [self.PATTERNS['arabic_assistant'], self.PATTERNS['english_assistant']]
                )
                
                if is_answer:
                    for pattern in [self.PATTERNS['arabic_assistant'], self.PATTERNS['english_assistant']]:
                        match = re.match(pattern, line, re.IGNORECASE)
                        if match:
                            answer = match.group(1).strip()
                            if len(answer) > 10:  # تجاهل الردود القصيرة جداً
                                raw_pairs.append({
                                    'instruction': current_instruction,
                                    'output': answer
                                })
                            current_instruction = None
                            break
                else:
                    # قد يكون جزء من إجابة متعددة الأسطر
                    pass
        
        # 2. معالجة وتنقية
        pairs = []
        for pair in raw_pairs:
            # تطبيع العربية
            if request.normalize_arabic:
                pair['instruction'] = self._normalize_arabic(pair['instruction'])
                pair['output'] = self._normalize_arabic(pair['output'])
            
            # تصحيح الترقيم
            if True:  # always apply
                pair['instruction'] = self._fix_punctuation(pair['instruction'])
                pair['output'] = self._fix_punctuation(pair['output'])
            
            # تقييم الجودة
            quality, score = self._assess_quality(pair, request.quality)
            
            # فلترة حسب الجودة
            if request.quality == ExtractionQuality.STRICT and quality != 'high':
                continue
            if request.quality == ExtractionQuality.BALANCED and quality == 'low':
                continue
            
            pairs.append(QAPair(
                instruction=pair['instruction'],
                output=pair['output'],
                quality=quality,
                score=score
            ))
        
        # 3. إزالة التكرار
        if request.remove_duplicates:
            pairs = self._remove_duplicates(pairs)
        
        # 4. فلترة الهلوسة (باستخدام LLM محلي)
        if request.remove_hallucinations and await ollama_client.is_available():
            pairs = await self._filter_hallucinations(pairs)
        
        # 5. حساب الإحصائيات
        total = len(raw_pairs)
        valid = len(pairs)
        removed = total - valid
        avg_quality = sum(p.score for p in pairs) / len(pairs) if pairs else 0
        
        return ExtractionResult(
            pairs=pairs,
            total=total,
            valid=valid,
            removed=removed,
            average_quality=round(avg_quality, 3)
        )
    
    def _normalize_arabic(self, text: str) -> str:
        """توحيد أشكال الحروف العربية"""
        replacements = {
            'إ': 'إ', 'أ': 'أ', 'آ': 'آ',
            'ة': 'ة', 'ى': 'ي',
            'ؤ': 'ؤ', 'ئ': 'ئ',
        }
        # توحيد الألف
        text = re.sub(r'[إأآا]', 'ا', text)
        # توحيد الياء
        text = re.sub(r'[ىي]', 'ي', text)
        # توحيد الهاء
        text = re.sub(r'[هة]', 'ة', text)
        # إزالة التشكيل
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
        return text
    
    def _fix_punctuation(self, text: str) -> str:
        """تصحيح الترقيم"""
        # إزالة المسافات قبل الترقيم
        text = re.sub(r'\s+([.,،;:!?])', r'\1', text)
        # إضافة مسافة بعد الترقيم
        text = re.sub(r'([.,،;:!?])([^\s])', r'\1 \2', text)
        # توحيد علامات الاستفهام
        text = text.replace('؟', '?')
        return text.strip()
    
    def _assess_quality(self, pair: Dict, quality_mode: ExtractionQuality) -> tuple:
        """تقييم جودة الزوج"""
        score = 0.5
        
        # طول الإجابة
        if len(pair['output']) > 50:
            score += 0.2
        if len(pair['output']) > 200:
            score += 0.1
        
        # وجود السؤال
        if any(c in pair['instruction'] for c in ['?', '؟', 'ما', 'كيف', 'لماذا', 'why', 'how']):
            score += 0.1
        
        # تجنب الردود القصيرة جداً
        if len(pair['output']) < 20:
            score -= 0.3
        
        # تحديد المستوى
        if score >= 0.8:
            return 'high', min(score, 1.0)
        elif score >= 0.5:
            return 'medium', score
        else:
            return 'low', score
    
    def _remove_duplicates(self, pairs: List[QAPair]) -> List[QAPair]:
        """إزالة التكرار بناءً على hash المحتوى"""
        seen = set()
        unique = []
        for pair in pairs:
            content_hash = hashlib.md5(
                (pair.instruction + pair.output).encode()
            ).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(pair)
        return unique
    
    async def _filter_hallucinations(self, pairs: List[QAPair]) -> List[QAPair]:
        """فلترة الهلوسة باستخدام LLM محلي"""
        filtered = []
        
        system_prompt = """أنت حكم جودة للبيانات التدريبية.
قيّم إذا كانت الإجابة تحتوي على هلوسة أو معلومات خاطئة.
أجب بـ "valid" أو "hallucination" فقط."""
        
        for pair in pairs:
            prompt = f"""السؤال: {pair.instruction}
الإجابة: {pair.output}

هل الإجابة صحيحة ومباشرة؟ أجب بـ valid أو hallucination:"""
            
            try:
                response = await ollama_client.generate(
                    model="qwen2.5:7b",
                    prompt=prompt,
                    system=system_prompt,
                    max_tokens=10,
                    stream=False
                )
                
                if "valid" in response.lower():
                    filtered.append(pair)
            except:
                # في حالة فشل Ollama، نحتفظ بالبيانات
                filtered.append(pair)
        
        return filtered
    
    async def save_to_jsonl(self, pairs: List[QAPair], filename: str) -> str:
        """حفظ الأزواج بصيغة JSONL"""
        output_path = settings.DATA_DIR / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                record = {
                    "instruction": pair.instruction,
                    "output": pair.output,
                    "quality": pair.quality,
                    "score": pair.score
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return str(output_path)
    
    async def save_to_csv(self, pairs: List[QAPair], filename: str) -> str:
        """حفظ بصيغة CSV"""
        import csv
        output_path = settings.DATA_DIR / filename
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['instruction', 'output', 'quality', 'score'])
            for pair in pairs:
                writer.writerow([pair.instruction, pair.output, pair.quality, pair.score])
        return str(output_path)

data_processor = DataProcessor()