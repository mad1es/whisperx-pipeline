# multimodal speech analysis

pipeline for extracting multimodal features (audio + text) from video/audio data.

## setup

```bash
python -m venv venv
source venv/bin/activate  # linux/mac
# or
venv\Scripts\activate  # windows

pip install -r requirements.txt
```

install ffmpeg:
- macos: `brew install ffmpeg`
- linux: `sudo apt-get install ffmpeg`
- windows: download from https://ffmpeg.org/download.html

build opensmile:
```bash
cd opensmile
mkdir build && cd build
cmake ..
make
```



place video files in `data/raw_videos/`.

## run

full pipeline:
```bash
python pipeline/run_full_pipeline.py
```

options:
- `--skip-audio` - skip audio extraction
- `--skip-transcription` - skip transcription
- `--skip-segmentation` - skip segmentation
- `--skip-features` - skip feature extraction
- `--skip-merge` - skip feature merging
- `--skip-training` - skip model training
- `--whisper-model` - whisper model (tiny/base/small/medium/large)
- `--whisper-language` - language (ru/en)
- `--whisper-device` - device (cpu/cuda)

visualization:
```bash
streamlit run visualization_app/app.py
```

## output

- `data/features/merged_features.csv` - merged features dataset
- `data/models/baseline_{model_type}.pkl` - trained model
- `data/results/` - plots and metrics
