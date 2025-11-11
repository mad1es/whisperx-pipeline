"""
Транскрипция аудио с использованием WhisperX для получения
точных таймстемпов на уровне слов и сегментов.

Автоматически определяет окружение (Mac/сервер) и оптимизирует параметры.
"""

import os
import json
import argparse
import platform
import psutil
from pathlib import Path
from tqdm import tqdm
import whisperx


def detect_environment():
    """
    Определяет тип окружения и возвращает оптимальные настройки.
    
    Returns:
        dict с настройками для окружения
    """
    is_mac = platform.system() == 'Darwin'
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    if is_mac or total_memory_gb < 16:
        return {
            'mode': 'lightweight',
            'default_model': 'medium',
            'batch_size': 4,
            'description': 'MacBook / ограниченная память'
        }
    else:
        return {
            'mode': 'server',
            'default_model': 'medium',
            'batch_size': 16,
            'description': 'Сервер / мощный компьютер'
        }


def transcribe_audio(
    audio_path: str,
    output_path: str,
    model_name: str = "medium",
    language: str = "ru",
    device: str = "cpu",
    compute_type: str = "int8",
    batch_size: int = None
) -> bool:
    """
    Транскрибирует аудио файл с помощью WhisperX.
    
    Args:
        audio_path: Путь к аудио файлу
        output_path: Путь для сохранения JSON с транскрипцией
        model_name: Название модели Whisper (tiny/base/small/medium/large)
        language: Язык аудио (ru/en и т.д.)
        device: Устройство для обработки (cpu/cuda)
        compute_type: Тип вычислений (int8/float16/float32)
        batch_size: Размер батча (None = автоопределение)
        
    Returns:
        True если успешно, False иначе
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if batch_size is None:
            env = detect_environment()
            batch_size = env['batch_size']
            print(f"Автоопределение: {env['description']}, batch_size={batch_size}")
        
        print(f"Загрузка модели {model_name}...")
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            language=language
        )
        
        print(f"Загрузка аудио...")
        audio = whisperx.load_audio(audio_path)
        
        print(f"Транскрипция с batch_size={batch_size}...")
        result = model.transcribe(audio, batch_size=batch_size)
        
        print(f"Выравнивание таймстемпов...")
        
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device=device,
            return_char_alignments=False
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при транскрипции {audio_path}: {e}")
        return False


def batch_transcribe(
    input_dir: str,
    output_dir: str,
    model_name: str = None,
    language: str = "ru",
    device: str = "cpu",
    batch_size: int = None,
    mode: str = None
) -> None:
    """
    Пакетная транскрипция всех аудио файлов.
    
    Args:
        input_dir: Директория с WAV файлами
        output_dir: Директория для сохранения JSON транскрипций
        model_name: Название модели Whisper (None = автоопределение)
        language: Язык аудио
        device: Устройство для обработки
        batch_size: Размер батча (None = автоопределение)
        mode: Режим работы ('lightweight' или 'server', None = автоопределение)
    """
    env = detect_environment()
    
    if mode == 'lightweight':
        env['mode'] = 'lightweight'
        env['default_model'] = 'medium'
        env['batch_size'] = 4
    elif mode == 'server':
        env['mode'] = 'server'
        env['default_model'] = 'medium'
        env['batch_size'] = 16
    
    if model_name is None:
        model_name = env['default_model']
    
    if batch_size is None:
        batch_size = env['batch_size']
    
    print(f"\n{'='*60}")
    print(f"Режим работы: {env['description']}")
    print(f"Модель: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    audio_files = list(input_path.glob('*.wav'))
    
    if not audio_files:
        print(f"Не найдено аудио файлов в {input_dir}")
        return
    
    print(f"Найдено {len(audio_files)} аудио файлов")
    
    success_count = 0
    for audio_file in tqdm(audio_files, desc="Транскрипция"):
        output_file = output_path / f"{audio_file.stem}.json"
        
        if transcribe_audio(
            str(audio_file),
            str(output_file),
            model_name=model_name,
            language=language,
            device=device,
            batch_size=batch_size
        ):
            success_count += 1
        else:
            print(f"Пропущен файл: {audio_file.name}")
    
    print(f"\nУспешно обработано: {success_count}/{len(audio_files)}")


def main():
    parser = argparse.ArgumentParser(
        description='Транскрипция аудио с WhisperX'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/audio_wav',
        help='Директория с WAV файлами'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/transcripts',
        help='Директория для сохранения JSON транскрипций'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='medium',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Модель Whisper для использования'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='ru',
        help='Язык аудио (ru/en и т.д.)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Устройство для обработки'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Размер батча (None = автоопределение по окружению)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['lightweight', 'server'],
        help='Режим работы: lightweight (MacBook) или server (мощный сервер)'
    )
    
    args = parser.parse_args()
    
    batch_transcribe(
        args.input_dir,
        args.output_dir,
        model_name=args.model,
        language=args.language,
        device=args.device,
        batch_size=args.batch_size,
        mode=args.mode
    )


if __name__ == '__main__':
    main()

