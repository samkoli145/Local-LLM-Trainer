.PHONY: build run stop clean logs shell test install dev

# بناء الصور مع تفعيل BuildKit
build:
	@echo "Building Docker images..."
	DOCKER_BUILDKIT=1 docker-compose -f docker/docker-compose.yml build --no-cache

# تشغيل كل الخدمات في الخلفية
run:
	@echo "Starting LocalTrainer..."
	docker-compose -f docker/docker-compose.yml up -d

# إيقاف الخدمات
stop:
	@echo "Stopping services..."
	docker-compose -f docker/docker-compose.yml down

# تنظيف كامل
clean:
	@echo "Cleaning up (including volumes)..."
	docker-compose -f docker/docker-compose.yml down -v --rmi all

# عرض السجلات
logs:
	docker-compose -f docker/docker-compose.yml logs -f app

# الدخول إلى حاوية التطبيق
shell:
	docker-compose -f docker/docker-compose.yml exec app /bin/bash

# تشغيل الاختبارات
test:
	python -m pytest tests/ -v --cov=backend

# تثبيت المتطلبات محلياً
install:
	pip install -r requirements.txt

# تشغيل التطوير المحلي
dev:
	@echo "Starting LocalTrainer in dev mode..."
	python -m backend.main
