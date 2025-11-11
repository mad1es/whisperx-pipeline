# Как добавлять новые видео в пайплайн

## Быстрая инструкция

### Шаг 1: Добавьте новые видео
Поместите новые видео файлы в `data/raw_videos/`

### Шаг 2: Обновите metadata.csv

**Рекомендуемый вариант: добавьте все видео (старые + новые)**

```csv
id,label
001_2025-01-27,1
014_2025.03.06,0
003_new_video,0
004_new_video,1
```

**Или только новые (система автоматически сохранит старые метаданные):**

```csv
id,label
003_new_video,0
004_new_video,1
```

### Шаг 3: Запустите пайплайн

```bash
python pipeline/run_full_pipeline.py --skip-training
```

## Как это работает

### Автоматическое объединение данных

1. **Сегменты**: Новые сегменты добавляются к существующим
2. **Признаки openSMILE**: Новые признаки добавляются к существующим  
3. **Метаданные**: Система автоматически создает `data/features/metadata_all.csv` со всеми видео
4. **Объединенный датасет**: Новые данные добавляются к существующим в `merged_features.csv`

### Перезапись видео

Если видео уже обработано и вы запустите пайплайн снова:
- Видео будет **перезаписано** (обновлено) с новыми данными
- Это полезно, если вы исправили транскрипцию или изменили метаданные

### Просмотр в Streamlit

Streamlit автоматически видит **все обработанные видео** из `merged_features.csv`:
- Старые видео (обработанные ранее)
- Новые видео (только что обработанные)

## Примеры

### Пример 1: Добавить одно новое видео

1. Добавьте `005_new.mp4` в `data/raw_videos/`
2. В `data/metadata.csv` добавьте:
   ```csv
   id,label
   001_2025-01-27,1
   014_2025.03.06,0
   005_new,1
   ```
3. Запустите: `python pipeline/run_full_pipeline.py --skip-training`
4. В Streamlit увидите все 3 видео

### Пример 2: Добавить несколько новых видео

1. Добавьте `006_new.mp4`, `007_new.mp4` в `data/raw_videos/`
2. В `data/metadata.csv` добавьте:
   ```csv
   id,label
   001_2025-01-27,1
   014_2025.03.06,0
   006_new,0
   007_new,1
   ```
3. Запустите: `python pipeline/run_full_pipeline.py --skip-training`
4. В Streamlit увидите все 4 видео

### Пример 3: Изменить label существующего видео

1. В `data/metadata.csv` измените label:
   ```csv
   id,label
   001_2025-01-27,0
   014_2025.03.06,1
   ```
2. Запустите только merge (без переобработки):
   ```bash
   python pipeline/merge_features.py \
       --segments-metadata data/segments/segments_metadata.csv \
       --opensmile-features data/features/opensmile_features.csv \
       --metadata data/metadata.csv \
       --output data/features/merged_features.csv \
       --language ru
   ```
3. Label обновится в `merged_features.csv`

## Важные замечания

### Имена файлов

- Имя файла должно совпадать с `id` в metadata.csv
- Пример: `001_2025-01-27.mp4` → `id: 001_2025-01-27`
- Расширение `.mp4` не указывается в metadata.csv

### Label

- `0` - контрольная группа (обычные люди)
- `1` - группа с суицидальными намерениями

### Файлы метаданных

- `data/metadata.csv` - ваш рабочий файл (обновляйте его)
- `data/features/metadata_all.csv` - автоматически создаваемый файл со всеми видео (не редактируйте вручную)

## Проверка результата

После обработки проверьте:

```bash
# Количество уникальных видео
python3 -c "import pandas as pd; df = pd.read_csv('data/features/merged_features.csv'); print('Видео:', df['file_id'].unique()); print('Всего:', len(df['file_id'].unique()))"

# Количество сегментов
python3 -c "import pandas as pd; df = pd.read_csv('data/features/merged_features.csv'); print('Всего сегментов:', len(df))"
```

## Удаление видео

Чтобы удалить видео из датасета:

1. Удалите файл из `data/raw_videos/`
2. Удалите строку из `data/metadata.csv`
3. Удалите папку из `data/segments/` (опционально)
4. Перезапустите merge (данные автоматически обновятся)

Или вручную отредактируйте `merged_features.csv`, удалив строки с нужным `file_id`.

