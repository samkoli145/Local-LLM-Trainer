"""
LocalTrainer Backend v3.0 - FastAPI Application
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import asyncio
import time

from backend.config import settings
from backend.schemas import (
    TrainingConfig, ChatRequest, ExtractionRequest,
    MergeRequest, DistillationRequest, TerminalCommand,
    CloudTrainingRequest, CloudValidationResult
)
from backend.ollama_client import ollama_client
from backend.lmstudio_client import lmstudio_client
from backend.trainer import training_engine
from backend.data_processor import data_processor
from backend.model_manager import model_manager
from backend.websocket_manager import ws_manager
from backend.audit_log import audit_log

app = FastAPI(
    title="LocalTrainer API",
    version="3.0.0",
    description="منصة تدريب نماذج LLM محلية"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
frontend_path = settings.BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="frontend")

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return index_path.read_text(encoding='utf-8')
    return "<h1>LocalTrainer Backend</h1><p>Frontend not found</p>"

# ============ Health & Info ============

@app.get("/api/health")
async def health_check():
    ollama_ok = await ollama_client.is_available()
    lmstudio_ok = await lmstudio_client.is_available()
    
    return {
        "status": "ok",
        "version": "3.0.0",
        "ollama": ollama_ok,
        "lm_studio": lmstudio_ok,
        "active_jobs": len([j for j in training_engine.active_jobs.values() if j.status == "running"])
    }

@app.get("/api/check-libraries")
async def check_libraries():
    """فحص المكتبات المثبتة"""
    result = {}
    
    libraries = {
        'unsloth': 'unsloth',
        'torch': 'torch',
        'transformers': 'transformers',
        'peft': 'peft',
        'trl': 'trl',
        'datasets': 'datasets',
        'bitsandbytes': 'bitsandbytes',
        'accelerate': 'accelerate',
    }
    
    for key, module in libraries.items():
        try:
            __import__(module)
            result[key] = 'installed'
        except ImportError:
            result[key] = 'missing'
    
    # Ollama
    result['ollama'] = 'installed' if await ollama_client.is_available() else 'missing'
    result['lmstudio'] = 'installed' if await lmstudio_client.is_available() else 'missing'
    
    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name(0)
            result['vram_gb'] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
        else:
            result['gpu'] = 'CPU only'
    except:
        result['gpu'] = 'unknown'
    
    return result

# ============ Ollama Endpoints ============

@app.get("/api/ollama/models")
async def list_ollama_models():
    """قائمة نماذج Ollama"""
    models = await ollama_client.list_models()
    return {"models": models}

@app.post("/api/ollama/chat")
async def ollama_chat(request: ChatRequest):
    """محادثة عبر Ollama"""
    audit_log.log("ollama_chat", {"model": request.model, "messages": len(request.messages)})
    
    response = await ollama_client.chat(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream
    )
    
    return response

@app.post("/api/ollama/generate")
async def ollama_generate(request: dict):
    """توليد نص عبر Ollama"""
    response = await ollama_client.generate(
        model=request.get('model'),
        prompt=request.get('prompt'),
        system=request.get('system'),
        temperature=request.get('temperature', 0.7),
        max_tokens=request.get('max_tokens', 512),
        stream=request.get('stream', False)
    )
    return response

@app.post("/api/ollama/pull")
async def pull_model(request: dict):
    """تحميل نموذج من Ollama registry"""
    model_name = request.get('model')
    if not model_name:
        raise HTTPException(400, "Model name required")
    
    async def stream_pull():
        async for status in ollama_client.pull_model(model_name):
            yield f"data: {status}\n\n"
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(stream_pull(), media_type="text/event-stream")

# ============ LM Studio Endpoints ============

@app.get("/api/lmstudio/models")
async def list_lmstudio_models():
    models = await lmstudio_client.list_models()
    return {"models": models}

@app.post("/api/lmstudio/chat")
async def lmstudio_chat(request: ChatRequest):
    response = await lmstudio_client.chat(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream
    )
    return response

# ============ Training Endpoints ============

@app.post("/api/train/start")
async def start_training(config: TrainingConfig):
    """بدء تدريب جديد"""
    if not config.dataset_path:
        raise HTTPException(400, "dataset_path is required")
    
    job_id = await training_engine.start_training(
        config=config,
        dataset_path=config.dataset_path
    )
    
    audit_log.log("training_started", {
        "job_id": job_id,
        "model": config.model,
        "method": config.method.value,
        "epochs": config.epochs
    })
    
    return {"job_id": job_id, "status": "started"}

@app.get("/api/train/status/{job_id}")
async def get_training_status(job_id: str):
    status = training_engine.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status

@app.post("/api/train/stop/{job_id}")
async def stop_training(job_id: str):
    success = await training_engine.stop_training(job_id)
    if not success:
        raise HTTPException(404, "Job not found or not running")
    return {"status": "stopped"}

@app.get("/api/train/jobs")
async def list_training_jobs():
    return training_engine.list_jobs()

# ============ Data Processing ============

@app.get("/api/data/files")
async def list_data_files():
    """قائمة ملفات البيانات المتاحة"""
    files = []
    if settings.DATA_DIR.exists():
        for p in sorted(settings.DATA_DIR.iterdir()):
            if p.is_file():
                size_mb = p.stat().st_size / 1e6
                lines = 0
                if p.suffix == ".jsonl":
                    try:
                        with open(p, "r") as f:
                            lines = sum(1 for _ in f)
                    except Exception:
                        pass
                files.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(size_mb, 1),
                    "lines": lines
                })
    return {"files": files}

@app.post("/api/data/extract")
async def extract_data(request: ExtractionRequest):
    """استخلاص أزواج Q&A"""
    result = await data_processor.extract_pairs(request)
    
    audit_log.log("data_extracted", {
        "total": result.total,
        "valid": result.valid,
        "removed": result.removed
    })
    
    return result

@app.post("/api/data/save")
async def save_data(request: dict):
    """حفظ البيانات بصيغة معينة"""
    pairs_data = request.get('pairs', [])
    format_type = request.get('format', 'jsonl')
    filename = request.get('filename', f"data_{int(time.time())}")
    
    from backend.schemas import QAPair
    pairs = [QAPair(**p) for p in pairs_data]
    
    if format_type == 'jsonl':
        path = await data_processor.save_to_jsonl(pairs, f"{filename}.jsonl")
    elif format_type == 'csv':
        path = await data_processor.save_to_csv(pairs, f"{filename}.csv")
    else:
        raise HTTPException(400, "Invalid format")
    
    return {"path": path}

# ============ Model Management ============

@app.get("/api/models")
async def list_models():
    return model_manager.list_models()

@app.get("/api/models/local")
async def list_local_models():
    """قائمة النماذج المحلية من القرص"""
    result = {"base": [], "trained": [], "checkpoints": []}

    # Base models
    if settings.BASE_MODELS_DIR.exists():
        for p in sorted(settings.BASE_MODELS_DIR.iterdir()):
            if p.is_dir():
                config_file = p / "config.json"
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
                info = {"name": p.name, "path": str(p), "size_mb": round(size_mb, 1)}
                if config_file.exists():
                    import json as _json
                    try:
                        cfg = _json.loads(config_file.read_text())
                        info["architectures"] = cfg.get("architectures", [])
                        info["model_type"] = cfg.get("model_type", "")
                        info["vocab_size"] = cfg.get("vocab_size", 0)
                        info["hidden_size"] = cfg.get("hidden_size", 0)
                        info["num_hidden_layers"] = cfg.get("num_hidden_layers", 0)
                    except Exception:
                        pass
                result["base"].append(info)

    # Trained models
    if settings.TRAINED_MODELS_DIR.exists():
        for p in sorted(settings.TRAINED_MODELS_DIR.iterdir()):
            if p.is_dir():
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
                has_checkpoint = any(c.is_dir() for c in p.iterdir() if c.is_dir())
                result["trained"].append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(size_mb, 1),
                    "has_checkpoint": has_checkpoint
                })

    # Checkpoints
    if settings.CHECKPOINTS_DIR.exists():
        for p in sorted(settings.CHECKPOINTS_DIR.iterdir()):
            if p.is_dir():
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
                result["checkpoints"].append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(size_mb, 1)
                })

    return result

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    model = model_manager.get_model(model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    return model

@app.post("/api/models/{model_id}/tags")
async def update_model_tags(model_id: str, tags: list):
    model_manager.update_tags(model_id, tags)
    audit_log.log("tags_updated", {"model_id": model_id, "tags": tags})
    return {"status": "ok"}

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    success = model_manager.delete_model(model_id)
    if not success:
        raise HTTPException(404, "Model not found")
    audit_log.log("model_deleted", {"model_id": model_id})
    return {"status": "deleted"}

@app.post("/api/models/merge")
async def merge_models(request: MergeRequest):
    import uuid
    job_id = f"merge_{uuid.uuid4().hex[:8]}"
    result = await model_manager.merge_models(request, job_id)
    return result

@app.post("/api/models/distill")
async def distill_models(request: DistillationRequest):
    import uuid
    job_id = f"distill_{uuid.uuid4().hex[:8]}"
    result = await model_manager.distill(request, job_id)
    return result

@app.post("/api/models/{model_id}/export-gguf")
async def export_gguf(model_id: str, request: dict = {}):
    quantization = request.get('quantization', 'q4_k_m')
    result = await model_manager.export_to_gguf(model_id, quantization)
    return result

@app.post("/api/models/{model_id}/import-ollama")
async def import_to_ollama(model_id: str, request: dict):
    ollama_name = request.get('name', model_id)
    result = await model_manager.import_to_ollama(model_id, ollama_name)
    return result

# ============ Cloud Training ============

@app.post("/api/cloud/validate-key")
async def validate_cloud_key(request: dict):
    """التحقق من API Key السحابي"""
    provider = request.get('provider')
    key = request.get('key')
    
    if not key or len(key) < 10:
        return CloudValidationResult(valid=False, provider=provider)
    
    # محاكاة التحقق (في الإنتاج: استدعاء API فعلي)
    return CloudValidationResult(
        valid=True,
        provider=provider,
        balance=100.0,
        available_gpus=["A100-80GB", "A100-40GB", "RTX-4090"]
    )

@app.post("/api/cloud/start")
async def start_cloud_training(request: CloudTrainingRequest):
    """بدء تدريب سحابي"""
    import uuid
    job_id = f"cloud_{uuid.uuid4().hex[:8]}"
    
    audit_log.log("cloud_training_started", {
        "job_id": job_id,
        "provider": request.provider,
        "gpu": request.gpu_type
    })
    
    return {
        "job_id": job_id,
        "status": "queued",
        "provider": request.provider,
        "estimated_cost": "$2.50/hour"
    }

# ============ Terminal ============

@app.post("/api/terminal/execute")
async def execute_terminal(request: TerminalCommand):
    """تنفيذ أمر طرفية (مقيد للأمان)"""
    cmd = request.command.strip()
    
    # فحص الأمان
    base_cmd = cmd.split()[0] if cmd else ""
    if base_cmd not in settings.ALLOWED_COMMANDS:
        return {
            "output": "",
            "error": f"Command not allowed: {base_cmd}",
            "return_code": 1,
            "duration_ms": 0
        }
    
    start_time = time.time()
    try:
        import subprocess
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(settings.BASE_DIR)
        )
        duration = (time.time() - start_time) * 1000
        
        return {
            "output": result.stdout,
            "error": result.stderr,
            "return_code": result.returncode,
            "duration_ms": round(duration, 2)
        }
    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": "Command timed out (30s)",
            "return_code": -1,
            "duration_ms": 30000
        }
    except Exception as e:
        return {
            "output": "",
            "error": str(e),
            "return_code": -1,
            "duration_ms": 0
        }

# ============ WebSocket ============

@app.websocket("/ws/training/{job_id}")
async def training_websocket(websocket: WebSocket, job_id: str):
    """WebSocket لتلقي تحديثات التدريب"""
    await ws_manager.join_room(websocket, job_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "STOP":
                await training_engine.stop_training(job_id)
                break
            elif data == "PING":
                await websocket.send_json({"type": "PONG"})
            else:
                try:
                    msg = json.loads(data)
                    if msg.get('type') == 'get_status':
                        status = training_engine.get_status(job_id)
                        if status:
                            await websocket.send_json({
                                "type": "status",
                                "data": status.dict()
                            })
                except json.JSONDecodeError:
                    pass
    except WebSocketDisconnect:
        await ws_manager.leave_room(websocket)

@app.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    """WebSocket للمحادثة المباشرة (streaming)"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            model = data.get('model', 'qwen2.5:7b')
            messages = data.get('messages', [])
            temperature = data.get('temperature', 0.7)
            
            from backend.schemas import ChatMessage
            chat_messages = [ChatMessage(**m) for m in messages]
            
            # إرسال streaming
            async for token in ollama_client.chat(
                model=model,
                messages=chat_messages,
                temperature=temperature,
                stream=True
            ):
                await websocket.send_json({
                    "type": "token",
                    "content": token
                })
            
            await websocket.send_json({"type": "done"})
            
    except WebSocketDisconnect:
        pass

# ============ Audit Log ============

@app.get("/api/audit")
async def get_audit_log(limit: int = 50):
    return audit_log.get_recent(limit)

# ============ Startup/Shutdown ============

@app.on_event("startup")
async def startup_event():
    print("🚀 LocalTrainer Backend v3.0 starting...")
    print(f"📁 Data dir: {settings.DATA_DIR}")
    print(f"📁 Models dir: {settings.MODELS_DIR}")
    print(f"🔗 Ollama: {settings.OLLAMA_BASE_URL}")
    
    # فحص توفر Ollama
    if await ollama_client.is_available():
        models = await ollama_client.list_models()
        print(f"✅ Ollama connected - {len(models)} models available")
    else:
        print("⚠️ Ollama not available")

@app.on_event("shutdown")
async def shutdown_event():
    await ollama_client.close()
    await lmstudio_client.close()
    print("👋 LocalTrainer Backend shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )