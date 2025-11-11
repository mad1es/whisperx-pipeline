# Решение проблем

## Проблема: Ноутбук зависает, память заполняется

### Причины:
1. **WhisperX загружает несколько моделей одновременно:**
   - Основная модель Whisper (medium = 1.5GB)
   - Модель выравнивания (alignment model)
   - VAD модель (voice activity detection)
   - **Итого: ~3-4 GB в памяти**

2. **batch_size=16 слишком большой для CPU/MacBook**

3. Все модели загружаются в RAM одновременно

### ✅ РЕШЕНИЕ: Автоматическое определение окружения

Пайплайн теперь **автоматически определяет** тип окружения и оптимизирует параметры:

**На MacBook:**
- Использует модель `medium` (та же точность)
- batch_size = 4 (вместо 16) - меньше нагрузка на память
- Автоматически определяется при запуске

**На сервере:**
- Использует модель `medium` (лучшая точность)
- batch_size = 16 (быстрее обработка)
- Автоматически определяется при запуске

### Как использовать:

#### 1. Автоматический режим (рекомендуется):
```bash
# Просто запустите - система сама определит MacBook и оптимизирует
python pipeline/run_full_pipeline.py
```

#### 2. Принудительный режим lightweight (MacBook):
```bash
python pipeline/run_full_pipeline.py --mode lightweight
```

#### 3. Принудительный режим server (мощный компьютер):
```bash
python pipeline/run_full_pipeline.py --mode server
```

### Если процесс уже завис:

1. **Прервите процесс** (Ctrl+C)

2. **Запустите с lightweight режимом:**
   ```bash
   python pipeline/run_full_pipeline.py --mode lightweight --skip-audio
   ```

3. **Или используйте tiny модель для теста:**
   ```bash
   python pipeline/run_full_pipeline.py --whisper-model tiny --skip-audio
   ```

### Рекомендации по памяти:

- **tiny** - ~200-300 MB памяти
- **base** - ~400-500 MB памяти  
- **small** - ~800 MB - 1 GB памяти
- **medium** - ~2-3 GB памяти (по умолчанию, batch_size=4 на MacBook)
- **large** - ~4-5 GB памяти

**Примечание:** На MacBook используется `medium` модель с уменьшенным `batch_size=4` для экономии памяти при сохранении точности.

### Проверка использования памяти:

```bash
# macOS
vm_stat

# Linux
free -h

# Или в Python:
python -c "import psutil; print(f'Память: {psutil.virtual_memory().percent}%')"
```

---

## Медленная загрузка модели WhisperX

### Проблема
При первом запуске WhisperX скачивает модель (1-3 GB), что может занять много времени при медленном интернете.

### Решения

#### 1. Использовать более легкую модель для теста

```bash
# Вместо medium используйте small или base
python pipeline/run_full_pipeline.py --whisper-model small
```

Размеры моделей:
- `tiny` - ~75 MB (самая быстрая, но менее точная)
- `base` - ~150 MB
- `small` - ~500 MB
- `medium` - ~1.5 GB (по умолчанию на серверах)
- `large` - ~3 GB (самая точная, но медленная)

#### 2. Прервать и использовать другую модель

Если загрузка слишком медленная, нажмите `Ctrl+C` и запустите с меньшей моделью:

```bash
# Прервать текущий процесс (Ctrl+C)
# Затем запустить с small моделью
python pipeline/run_full_pipeline.py --whisper-model small --skip-audio
```

#### 3. Где хранятся модели

Модели WhisperX кэшируются в:
- macOS/Linux: `~/.cache/whisperx/`
- Windows: `%USERPROFILE%\.cache\whisperx\`

После первой загрузки модель будет использоваться из кэша.

### Рекомендации

- Для **быстрого теста** используйте `small` или `base`
- Для **производства** используйте `medium` или `large`
- Если интернет медленный, скачайте модель вручную или используйте VPN

---

## Другие проблемы

### Ошибка "CUDA out of memory"
Используйте CPU или уменьшите batch_size:
```bash
python pipeline/run_full_pipeline.py --whisper-device cpu --mode lightweight
```

### Ошибка "FFmpeg not found"
Установите FFmpeg:
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Ошибка "SMILExtract not found"
Убедитесь, что openSMILE собран:
```bash
cd opensmile/build
make
```
