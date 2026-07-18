"""
إعدادات التطبيق - LocalTrainer v3.0
جميع الثوابت والإعدادات المركزية هنا
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Settings:
    # المسارات
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    BASE_MODELS_DIR: Path = BASE_DIR / "models" / "base"
    TRAINED_MODELS_DIR: Path = BASE_DIR / "models" / "trained"
    CHECKPOINTS_DIR: Path = BASE_DIR / "checkpoints"
    LOGS_DIR: Path = BASE_DIR / "logs"
    
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: int = 300  # 5 دقائق
    
    # LM Studio (fallback)
    LM_STUDIO_BASE_URL: str = "http://localhost:1234/v1"
    
    # التدريب
    DEFAULT_BATCH_SIZE: int = 2  # آمن لـ 11GB VRAM
    DEFAULT_MAX_SEQ_LEN: int = 2048
    MAX_VRAM_GB: int = 11
    DEFAULT_LORA_RANK: int = 16
    DEFAULT_LORA_ALPHA: int = 32
    
    # الخادم
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # الأمان
    ALLOWED_COMMANDS: list = None
    
    def __post_init__(self):
        # إنشاء المجلدات تلقائياً
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.CHECKPOINTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # الأوامر المسموحة في Terminal
        if self.ALLOWED_COMMANDS is None:
            self.ALLOWED_COMMANDS = [
                'ollama', 'nvidia-smi', 'python', 'pip', 'ls', 'cat',
                'df', 'free', 'ps', 'top', 'htop', 'watch', 'tail',
                'grep', 'find', 'du', 'uname', 'lspci', 'nvcc'
            ]

# Instance واحد للاستخدام في كل التطبيق
settings = Settings()