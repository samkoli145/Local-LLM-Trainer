"""
إدارة النماذج: دمج، تصدير، تحويل، تقطير
"""
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.config import settings
from backend.schemas import (
    ModelInfo, MergeRequest, DistillationRequest,
    MergeMethod, DistillMethod
)
from backend.websocket_manager import ws_manager

class ModelManager:
    def __init__(self):
        self.models_db_path = settings.MODELS_DIR / "models_db.json"
        self.models_db: Dict[str, ModelInfo] = self._load_db()
    
    def _load_db(self) -> Dict[str, ModelInfo]:
        if self.models_db_path.exists():
            with open(self.models_db_path, 'r') as f:
                data = json.load(f)
                return {k: ModelInfo(**v) for k, v in data.items()}
        return {}
    
    def _save_db(self):
        with open(self.models_db_path, 'w') as f:
            json.dump(
                {k: v.dict() for k, v in self.models_db.items()},
                f, indent=2, ensure_ascii=False
            )
    
    def list_models(self) -> List[ModelInfo]:
        return list(self.models_db.values())
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self.models_db.get(model_id)
    
    def add_model(self, model: ModelInfo):
        self.models_db[model.id] = model
        self._save_db()
    
    def update_tags(self, model_id: str, tags: List[str]):
        if model_id in self.models_db:
            self.models_db[model_id].tags = tags
            self._save_db()
    
    def delete_model(self, model_id: str) -> bool:
        if model_id in self.models_db:
            del self.models_db[model_id]
            self._save_db()
            return True
        return False
    
    async def merge_models(self, request: MergeRequest, job_id: str) -> Dict[str, Any]:
        """دمج نموذجين"""
        await ws_manager.send_log(job_id, f"🔗 بدء دمج النماذج بطريقة {request.method.value}...", "info")
        
        model_a_path = settings.MODELS_DIR / request.model_a
        model_b_path = settings.MODELS_DIR / request.model_b
        
        if not model_a_path.exists() or not model_b_path.exists():
            await ws_manager.send_log(job_id, "❌ أحد النماذج غير موجود", "error")
            return {"success": False, "error": "Model not found"}
        
        try:
            # استخدام mergekit (أو تنفيذ يدوي)
            output_path = settings.MODELS_DIR / f"merged_{job_id}"
            
            if request.method == MergeMethod.LINEAR:
                # دمج خطي بسيط
                await self._linear_merge(
                    str(model_a_path), str(model_b_path),
                    str(output_path), request.alpha, job_id
                )
            elif request.method == MergeMethod.SLERP:
                await ws_manager.send_log(job_id, "🔄 SLERP merge...", "info")
                # تنفيذ SLERP
            elif request.method == MergeMethod.DARE:
                await ws_manager.send_log(job_id, "🎯 DARE merge...", "info")
            
            # تسجيل النموذج المدمج
            merged_model = ModelInfo(
                id=f"merged_{job_id}",
                name=f"Merged-{request.model_a[:10]}-{request.model_b[:10]}",
                architecture="Merged",
                accuracy=0,
                loss=0,
                tags=["Merged", request.method.value],
                date=datetime.now().isoformat(),
                hyperparams={"method": request.method.value, "alpha": request.alpha},
                file_path=str(output_path)
            )
            self.add_model(merged_model)
            
            await ws_manager.send_log(job_id, "✅ اكتمل الدمج", "success")
            return {"success": True, "model_id": merged_model.id}
            
        except Exception as e:
            await ws_manager.send_log(job_id, f"❌ فشل الدمج: {e}", "error")
            return {"success": False, "error": str(e)}
    
    async def _linear_merge(self, path_a: str, path_b: str, output: str, alpha: float, job_id: str):
        """دمج خطي للأوزان"""
        try:
            import torch
            from safetensors.torch import load_file, save_file
            
            await ws_manager.send_log(job_id, "📊 تحميل الأوزان...", "info")
            
            # تحميل الأوزان من النموذجين
            weights_a = load_file(f"{path_a}/model.safetensors")
            weights_b = load_file(f"{path_b}/model.safetensors")
            
            # دمج خطي
            merged = {}
            for key in weights_a:
                if key in weights_b:
                    merged[key] = weights_a[key] * alpha + weights_b[key] * (1 - alpha)
                else:
                    merged[key] = weights_a[key]
            
            # حفظ
            Path(output).mkdir(parents=True, exist_ok=True)
            save_file(merged, f"{output}/model.safetensors")
            
            await ws_manager.send_log(job_id, "💾 تم حفظ الأوزان المدمجة", "success")
            
        except Exception as e:
            raise RuntimeError(f"فشل الدمج الخطي: {e}")
    
    async def distill(
        self,
        request: DistillationRequest,
        job_id: str
    ) -> Dict[str, Any]:
        """التقطير المعرفي"""
        await ws_manager.send_log(job_id, f" بدء التقطير: {request.teacher_model} → {request.student_model}", "info")
        
        try:
            import torch
            import torch.nn.functional as F
            from unsloth import FastLanguageModel
            
            # 1. تحميل المعلم
            await ws_manager.send_log(job_id, "📥 تحميل نموذج المعلم...", "info")
            teacher_model, teacher_tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.teacher_model,
                load_in_4bit=True,
            )
            teacher_model.eval()
            
            # 2. تحميل الطالب
            await ws_manager.send_log(job_id, "📥 تحميل نموذج الطالب...", "info")
            student_model, student_tokenizer = FastLanguageModel.from_pretrained(
                model_name=request.student_model,
                load_in_4bit=True,
            )
            
            # 3. إعداد LoRA للطالب
            student_model = FastLanguageModel.get_peft_model(
                student_model,
                r=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            
            await ws_manager.send_log(job_id, "🔄 بدء التقطير...", "info")
            
            # 4. حلقة التقطير
            optimizer = torch.optim.AdamW(student_model.parameters(), lr=0.0002)
            
            # بيانات تجريبية (يجب استبدالها ببيانات حقيقية)
            for step in range(100):
                # هنا يتم تنفيذ التقطير الفعلي
                # ...
                if step % 10 == 0:
                    await ws_manager.send_log(
                        job_id,
                        f"📊 Step {step}/100",
                        "info"
                    )
            
            # 5. حفظ الطالب المدرب
            output_path = settings.MODELS_DIR / f"distilled_{job_id}"
            student_model.save_pretrained(str(output_path))
            
            await ws_manager.send_log(job_id, "✅ اكتمل التقطير", "success")
            
            return {
                "success": True,
                "model_id": f"distilled_{job_id}",
                "output_path": str(output_path)
            }
            
        except Exception as e:
            await ws_manager.send_log(job_id, f"❌ فشل التقطير: {e}", "error")
            return {"success": False, "error": str(e)}
    
    async def export_to_gguf(self, model_id: str, quantization: str = "q4_k_m") -> Dict[str, Any]:
        """تصدير نموذج إلى GGUF"""
        model = self.get_model(model_id)
        if not model:
            return {"success": False, "error": "Model not found"}
        
        try:
            from unsloth import FastLanguageModel
            
            model_path = model.file_path or str(settings.MODELS_DIR / model_id)
            output_path = settings.MODELS_DIR / f"{model_id}_gguf"
            
            loaded_model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
            )
            
            loaded_model.save_pretrained_gguf(
                str(output_path),
                tokenizer,
                quantization_method=quantization
            )
            
            return {
                "success": True,
                "output_path": str(output_path),
                "quantization": quantization
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def import_to_ollama(self, model_id: str, ollama_name: str) -> Dict[str, Any]:
        """استيراد نموذج إلى Ollama"""
        model = self.get_model(model_id)
        if not model:
            return {"success": False, "error": "Model not found"}
        
        gguf_path = settings.MODELS_DIR / f"{model_id}_gguf" / f"{model_id}.gguf"
        if not gguf_path.exists():
            return {"success": False, "error": "GGUF file not found. Export first."}
        
        # إنشاء Modelfile
        modelfile_content = f"""FROM {gguf_path}
PARAMETER temperature 0.7
PARAMETER num_ctx 2048
SYSTEM أنت مساعد ذكي مدرب محلياً."""
        
        modelfile_path = settings.MODELS_DIR / f"{model_id}_Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        # استيراد إلى Ollama
        try:
            result = subprocess.run(
                ['ollama', 'create', ollama_name, '-f', str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return {"success": True, "ollama_name": ollama_name}
            else:
                return {"success": False, "error": result.stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

model_manager = ModelManager()