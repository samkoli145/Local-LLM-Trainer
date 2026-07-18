"""
نماذج البيانات للـ API - validation صارم
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class TrainingMode(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"

class FineTuneMethod(str, Enum):
    QLORA = "qlora"
    LORA = "lora"
    FULL = "full"

class DistillMethod(str, Enum):
    ADAPTER_TRANSFER = "adapter_transfer"
    LOGITS_DISTILLATION = "logits_distillation"
    FEATURE_DISTILLATION = "feature_distillation"
    WEIGHT_MERGING = "weight_merging"

class MergeMethod(str, Enum):
    LINEAR = "linear"
    SLERP = "slerp"
    DARE = "dare"
    TIES = "ties"

class ExtractionStyle(str, Enum):
    QA = "qa"
    INSTRUCTION = "instruction"
    CHAT = "chat"
    COMPLETION = "completion"

class ExtractionQuality(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    LENIENT = "lenient"

# ============ Training Schemas ============

class TrainingConfig(BaseModel):
    model: str = Field(..., min_length=3, description="اسم النموذج الأساسي")
    mode: TrainingMode = TrainingMode.LOCAL
    method: FineTuneMethod = FineTuneMethod.QLORA
    learning_rate: float = Field(0.0002, gt=0, lt=0.1)
    batch_size: int = Field(2, ge=1, le=32)
    epochs: int = Field(3, ge=1, le=100)
    lora_rank: int = Field(16, ge=4, le=128)
    lora_alpha: int = Field(32, ge=8)
    max_seq_len: int = Field(2048, ge=256, le=8192)
    gradient_accumulation: int = Field(4, ge=1, le=32)
    weight_decay: float = Field(0.01, ge=0, lt=1)
    dropout: float = Field(0.05, ge=0, le=0.5)
    target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    dataset_path: Optional[str] = None
    cloud_api_key: Optional[str] = None
    cloud_provider: Optional[str] = None
    cloud_gpu: Optional[str] = None
    
    @validator('lora_alpha')
    def alpha_must_be_multiple_of_rank(cls, v, values):
        if 'lora_rank' in values and v < values['lora_rank']:
            raise ValueError('lora_alpha يجب أن يكون >= lora_rank')
        return v

class TrainingStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "stopped"]
    progress: float = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    current_accuracy: Optional[float] = None
    started_at: Optional[datetime] = None
    error: Optional[str] = None

# ============ Chat Schemas ============

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: List[ChatMessage]
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(512, ge=1, le=4096)
    stream: bool = True

class ChatResponse(BaseModel):
    model: str
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

# ============ Data Processing Schemas ============

class ExtractionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=1000000)
    style: ExtractionStyle = ExtractionStyle.QA
    quality: ExtractionQuality = ExtractionQuality.BALANCED
    remove_duplicates: bool = True
    remove_hallucinations: bool = True
    normalize_arabic: bool = True

class QAPair(BaseModel):
    instruction: str
    output: str
    quality: Literal["high", "medium", "low"]
    score: float = Field(ge=0, le=1)
    source: Optional[str] = None

class ExtractionResult(BaseModel):
    pairs: List[QAPair]
    total: int
    valid: int
    removed: int
    average_quality: float

# ============ Model Management Schemas ============

class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    accuracy: float
    loss: float
    tags: List[str] = []
    date: str
    hyperparams: Dict[str, Any]
    loss_history: List[float] = []
    file_path: Optional[str] = None
    size_mb: Optional[float] = None

class MergeRequest(BaseModel):
    model_config = {'protected_namespaces': ()}

    model_a: str
    model_b: str
    method: MergeMethod = MergeMethod.LINEAR
    alpha: float = Field(0.5, ge=0, le=1)

class DistillationRequest(BaseModel):
    teacher_model: str
    student_model: str
    method: DistillMethod = DistillMethod.LOGITS_DISTILLATION
    temperature: float = Field(2.0, ge=0.1, le=10)
    teacher_weight: float = Field(0.7, ge=0, le=1)

# ============ Cloud Schemas ============

class CloudTrainingRequest(BaseModel):
    provider: Literal["runpod", "lambda", "vast", "modal"]
    api_key: str = Field(..., min_length=10)
    gpu_type: str = "A100-80GB"
    gpu_count: int = Field(1, ge=1, le=8)
    training_config: TrainingConfig

class CloudValidationResult(BaseModel):
    valid: bool
    provider: str
    balance: Optional[float] = None
    available_gpus: List[str] = []

# ============ Terminal Schemas ============

class TerminalCommand(BaseModel):
    command: str = Field(..., min_length=1, max_length=500)

class TerminalResponse(BaseModel):
    output: str
    error: str = ""
    return_code: int
    duration_ms: float