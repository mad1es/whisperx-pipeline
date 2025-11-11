"""
Сегментация аудио на основе транскрипций WhisperX.

Создает сегменты речи длительностью 2-5 секунд, выровненные по сегментам Whisper,
и сохраняет их как отдельные WAV файлы с соответствующими метаданными.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
import pandas as pd


def load_whisper_transcript(transcript_path: str) -> dict:
    """
    Загружает транскрипцию WhisperX из JSON файла.
    
    Args:
        transcript_path: Путь к JSON файлу с транскрипцией
        
    Returns:
        Словарь с данными транскрипции
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_segments_from_transcript(
    transcript: dict,
    audio_path: str,
    output_dir: str,
    min_segment_duration: float = 2.0,
    max_segment_duration: float = 5.0,
    overlap: float = 0.0
) -> pd.DataFrame:
    """
    Создает сегменты аудио на основе транскрипции.
    
    Args:
        transcript: Словарь с данными транскрипции WhisperX
        audio_path: Путь к исходному аудио файлу
        output_dir: Директория для сохранения сегментов
        min_segment_duration: Минимальная длительность сегмента (сек)
        max_segment_duration: Максимальная длительность сегмента (сек)
        overlap: Перекрытие между сегментами (сек)
        
    Returns:
        DataFrame с метаданными сегментов
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio = AudioSegment.from_wav(audio_path)
    file_id = Path(audio_path).stem
    
    segments_metadata = []
    
    if 'segments' not in transcript:
        print(f"Предупреждение: нет сегментов в {audio_path}")
        return pd.DataFrame()
    
    segment_idx = 0
    for seg in transcript['segments']:
        start_time = seg['start']
        end_time = seg['end']
        text = seg.get('text', '').strip()
        
        duration = end_time - start_time
        
        if duration < min_segment_duration:
            continue
        
        if duration > max_segment_duration:
            num_subsegments = int(duration / max_segment_duration) + 1
            subsegment_duration = duration / num_subsegments
            
            for i in range(num_subsegments):
                sub_start = start_time + i * subsegment_duration
                sub_end = min(start_time + (i + 1) * subsegment_duration, end_time)
                
                segment_wav_path = os.path.join(
                    output_dir,
                    f"segment_{segment_idx:04d}.wav"
                )
                
                start_ms = int(sub_start * 1000)
                end_ms = int(sub_end * 1000)
                segment_audio = audio[start_ms:end_ms]
                segment_audio.export(segment_wav_path, format="wav")
                
                words_in_segment = []
                word_confidences = []
                
                if 'words' in seg:
                    for word in seg['words']:
                        word_start = word.get('start', 0)
                        word_end = word.get('end', 0)
                        if sub_start <= word_start < sub_end or sub_start < word_end <= sub_end:
                            words_in_segment.append(word.get('word', ''))
                            word_confidences.append(word.get('score', 0.0))
                
                segment_text = ' '.join(words_in_segment) if words_in_segment else text
                
                segments_metadata.append({
                    'file_id': file_id,
                    'segment_id': f"{file_id}_segment_{segment_idx:04d}",
                    'start': sub_start,
                    'end': sub_end,
                    'duration': sub_end - sub_start,
                    'text': segment_text,
                    'asr_conf_mean': sum(word_confidences) / len(word_confidences) if word_confidences else 0.0,
                    'asr_conf_std': pd.Series(word_confidences).std() if len(word_confidences) > 1 else 0.0,
                    'word_count': len(words_in_segment),
                    'segment_path': segment_wav_path
                })
                
                segment_idx += 1
        else:
            segment_wav_path = os.path.join(
                output_dir,
                f"segment_{segment_idx:04d}.wav"
            )
            
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            segment_audio = audio[start_ms:end_ms]
            segment_audio.export(segment_wav_path, format="wav")
            
            words = seg.get('words', [])
            word_confidences = [w.get('score', 0.0) for w in words]
            
            segments_metadata.append({
                'file_id': file_id,
                'segment_id': f"{file_id}_segment_{segment_idx:04d}",
                'start': start_time,
                'end': end_time,
                'duration': duration,
                'text': text,
                'asr_conf_mean': sum(word_confidences) / len(word_confidences) if word_confidences else 0.0,
                'asr_conf_std': pd.Series(word_confidences).std() if len(word_confidences) > 1 else 0.0,
                'word_count': len(words),
                'segment_path': segment_wav_path
            })
            
            segment_idx += 1
    
    return pd.DataFrame(segments_metadata)


def batch_segment_audio(
    audio_dir: str,
    transcript_dir: str,
    output_base_dir: str,
    min_segment_duration: float = 2.0,
    max_segment_duration: float = 5.0
) -> None:
    """
    Пакетная сегментация всех аудио файлов.
    
    Args:
        audio_dir: Директория с WAV файлами
        transcript_dir: Директория с JSON транскрипциями
        output_base_dir: Базовая директория для сохранения сегментов
        min_segment_duration: Минимальная длительность сегмента
        max_segment_duration: Максимальная длительность сегмента
    """
    audio_path = Path(audio_dir)
    transcript_path = Path(transcript_dir)
    output_path = Path(output_base_dir)
    
    audio_files = list(audio_path.glob('*.wav'))
    
    if not audio_files:
        print(f"Не найдено аудио файлов в {audio_dir}")
        return
    
    all_segments = []
    
    for audio_file in tqdm(audio_files, desc="Сегментация"):
        transcript_file = transcript_path / f"{audio_file.stem}.json"
        
        if not transcript_file.exists():
            print(f"Пропущен {audio_file.name}: нет транскрипции")
            continue
        
        transcript = load_whisper_transcript(str(transcript_file))
        segment_output_dir = output_path / audio_file.stem
        
        segments_df = create_segments_from_transcript(
            transcript,
            str(audio_file),
            str(segment_output_dir),
            min_segment_duration=min_segment_duration,
            max_segment_duration=max_segment_duration
        )
        
        if not segments_df.empty:
            all_segments.append(segments_df)
    
    if all_segments:
        combined_df = pd.concat(all_segments, ignore_index=True)
        metadata_path = output_path / 'segments_metadata.csv'
        
        if metadata_path.exists():
            existing_df = pd.read_csv(metadata_path)
            existing_file_ids = set(existing_df['file_id'].unique())
            new_file_ids = set(combined_df['file_id'].unique())
            
            if existing_file_ids & new_file_ids:
                print(f"Предупреждение: файлы {existing_file_ids & new_file_ids} уже есть в метаданных. Перезаписываю.")
                combined_df = pd.concat([existing_df[~existing_df['file_id'].isin(new_file_ids)], combined_df], ignore_index=True)
            else:
                combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
                print(f"Добавлено {len(new_file_ids)} новых видео к существующим {len(existing_file_ids)}")
        
        combined_df.to_csv(metadata_path, index=False, encoding='utf-8')
        print(f"\nВсего сегментов: {len(combined_df)}")
        print(f"Уникальных видео: {combined_df['file_id'].nunique()}")
        print(f"Метаданные сохранены в {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Сегментация аудио на основе транскрипций'
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='data/audio_wav',
        help='Директория с WAV файлами'
    )
    parser.add_argument(
        '--transcript-dir',
        type=str,
        default='data/transcripts',
        help='Директория с JSON транскрипциями'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/segments',
        help='Директория для сохранения сегментов'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=2.0,
        help='Минимальная длительность сегмента (сек)'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=5.0,
        help='Максимальная длительность сегмента (сек)'
    )
    
    args = parser.parse_args()
    
    batch_segment_audio(
        args.audio_dir,
        args.transcript_dir,
        args.output_dir,
        min_segment_duration=args.min_duration,
        max_segment_duration=args.max_duration
    )


if __name__ == '__main__':
    main()

