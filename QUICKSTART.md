# Быстрый старт

## 🎬 Обработка одного видео (самый простой способ)

1. **Поместите видео** в `data/raw_videos/` (например, `001_2025-01-27.mp4`)

2. **Создайте `data/metadata.csv`:**
   ```csv
   id,label
   001_2025-01-27,0
   ```

3. **Запустите пайплайн:**
   ```bash
   python pipeline/run_full_pipeline.py
   ```

4. **Посмотрите результаты:**
   ```bash
   # Интерактивная визуализация
   streamlit run visualization_app/app.py
   
   # Или проверьте файлы вручную:
   # - data/features/merged_features.csv - все признаки
   # - data/results/ - графики модели (если обучение не пропущено)
   ```

**Время обработки:** ~5-15 минут на одно видео (зависит от длины и устройства)

---

## Минимальная настройка

1. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

2. **Проверьте FFmpeg:**
```bash
ffmpeg -version
```

3. **Подготовьте данные:**
   - Поместите видео файлы в `data/raw_videos/`
   - Создайте `data/metadata.csv` (скопируйте из `data/metadata.csv.example`)

4. **Запустите полный пайплайн:**
```bash
python pipeline/run_full_pipeline.py
```

## Пошаговая обработка одного видео

Если хотите обработать одно видео для тестирования:

```bash
# 1. Извлечение аудио
python pipeline/extract_audio.py --input-dir data/raw_videos --output-dir data/audio_wav

# 2. Транскрипция (может занять время)
python pipeline/transcribe_whisperx.py --input-dir data/audio_wav --output-dir data/transcripts --model medium --language ru

# 3. Сегментация
python pipeline/segment_audio.py --audio-dir data/audio_wav --transcript-dir data/transcripts --output-dir data/segments

# 4. Извлечение признаков openSMILE
python pipeline/extract_opensmile_features.py --segments-dir data/segments --output-dir data/features

# 5. Объединение признаков
python pipeline/merge_features.py --segments-metadata data/segments/segments_metadata.csv --opensmile-features data/features/opensmile_features.csv --output data/features/merged_features.csv

# 6. Визуализация
streamlit run visualization_app/app.py
```

## Структура metadata.csv

```csv
id,label
001_2025-01-27,0
002_2025-01-28,1
```

Где:
- `id` - имя файла без расширения (например, `001_2025-01-27` для `001_2025-01-27.mp4`)
- `label` - `0` для контрольной группы, `1` для группы с суицидальными намерениями

## Примечания

- WhisperX автоматически скачает модель при первом запуске
- Для GPU ускорения используйте `--device cuda` в транскрипции
- Обработка одного видео может занять 5-15 минут в зависимости от длины и устройства

