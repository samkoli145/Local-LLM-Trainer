"""
سجل التدقيق - تتبع كل العمليات المهمة
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from backend.config import settings

class AuditLog:
    def __init__(self):
        self.log_path = settings.LOGS_DIR / "audit.jsonl"
    
    def log(self, action: str, details: Dict[str, Any], user: str = "local"):
        """تسجيل حدث"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "action": action,
            "details": details
        }
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def get_recent(self, limit: int = 50) -> list:
        """آخر الأحداث"""
        if not self.log_path.exists():
            return []
        
        entries = []
        with open(self.log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        return entries[-limit:]

audit_log = AuditLog()