# Детектор Латука

Нейросеть для распознавания Латука на фотографиях.

## 🧠 Модель

- Архитектура: YOLO11
- Обучалась на MacBook M2
- Размер модели: ~6 МБ

## 📦 Использование

```python
from ultralytics import YOLO

# Загрузить модель
model = YOLO('best.pt')

# Распознать Латука на фото
results = model('photo.jpg')
results[0].show()
