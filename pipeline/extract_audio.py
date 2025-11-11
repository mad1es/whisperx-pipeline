"""
Извлечение аудио из видео файлов с нормализацией.

Конвертирует видео в моно WAV файлы с частотой дискретизации 16kHz
и нормализует громкость до -20 dBFS для консистентности.
"""

import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_audio_from_video(
    video_path: str,
    output_path: str,
    sample_rate: int = 16000,
    normalize_db: float = -20.0
) -> bool:
    """
    Извлекает аудио из видео и нормализует его.
    
    Args:
        video_path: Путь к входному видео файлу
        output_path: Путь для сохранения WAV файла
        sample_rate: Частота дискретизации (по умолчанию 16kHz)
        normalize_db: Уровень нормализации в dBFS
        
    Returns:
        True если успешно, False иначе
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # Без видео
            '-ac', '1',  # Моно
            '-ar', str(sample_rate),  # Частота дискретизации
            '-af', f'loudnorm=I={normalize_db}:TP=-1.5:LRA=11',  # Нормализация
            '-y',  # Перезаписать если существует
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при обработке {video_path}: {e.stderr}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False


def batch_extract_audio(
    input_dir: str,
    output_dir: str,
    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')
) -> None:
    """
    Пакетная обработка всех видео файлов в директории.
    
    Args:
        input_dir: Директория с видео файлами
        output_dir: Директория для сохранения WAV файлов
        video_extensions: Расширения видео файлов для обработки
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f'*{ext}'))
    
    if not video_files:
        print(f"Не найдено видео файлов в {input_dir}")
        return
    
    print(f"Найдено {len(video_files)} видео файлов")
    
    success_count = 0
    for video_file in tqdm(video_files, desc="Извлечение аудио"):
        output_file = output_path / f"{video_file.stem}.wav"
        
        if extract_audio_from_video(str(video_file), str(output_file)):
            success_count += 1
        else:
            print(f"Пропущен файл: {video_file.name}")
    
    print(f"\nУспешно обработано: {success_count}/{len(video_files)}")


def main():
    parser = argparse.ArgumentParser(
        description='Извлечение аудио из видео файлов'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw_videos',
        help='Директория с видео файлами'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/audio_wav',
        help='Директория для сохранения WAV файлов'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Частота дискретизации (по умолчанию 16000)'
    )
    
    args = parser.parse_args()
    
    batch_extract_audio(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

