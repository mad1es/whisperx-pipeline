"""
Объединение аудио и текстовых признаков в единый датасет.

Объединяет признаки openSMILE с транскрипциями и метаданными сегментов,
добавляет производные признаки (speech rate, sentiment и т.д.).
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_speech_rate(text: str, duration: float) -> float:
    """
    Вычисляет скорость речи (слов в секунду).
    
    Args:
        text: Текст сегмента
        duration: Длительность сегмента в секундах
        
    Returns:
        Скорость речи (слов/сек)
    """
    if duration <= 0:
        return 0.0
    
    words = text.split()
    return len(words) / duration


def calculate_articulation_rate(text: str, duration: float, pause_ratio: float = 0.0) -> float:
    """
    Вычисляет артикуляционную скорость (слов в секунду активной речи).
    
    Args:
        text: Текст сегмента
        duration: Длительность сегмента в секундах
        pause_ratio: Доля пауз в сегменте
        
    Returns:
        Артикуляционная скорость
    """
    if duration <= 0:
        return 0.0
    
    words = text.split()
    active_duration = duration * (1 - pause_ratio)
    
    if active_duration <= 0:
        return 0.0
    
    return len(words) / active_duration


def calculate_type_token_ratio(text: str) -> float:
    """
    Вычисляет Type-Token Ratio (лексическое разнообразие).
    
    Args:
        text: Текст сегмента
        
    Returns:
        TTR (отношение уникальных слов к общему количеству)
    """
    words = text.lower().split()
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    return unique_words / len(words)


def count_pronouns(text: str, language: str = 'ru') -> int:
    """
    Подсчитывает количество местоимений в тексте.
    
    Args:
        text: Текст сегмента
        language: Язык текста
        
    Returns:
        Количество местоимений
    """
    if language == 'ru':
        pronouns = [
            'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
            'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их',
            'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
            'мной', 'тобой', 'им', 'ей', 'нами', 'вами', 'ими',
            'мой', 'твой', 'его', 'её', 'наш', 'ваш', 'их'
        ]
    else:
        pronouns = [
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'our', 'their'
        ]
    
    words = text.lower().split()
    return sum(1 for word in words if word in pronouns)


def count_negations(text: str, language: str = 'ru') -> int:
    """
    Подсчитывает количество отрицаний в тексте.
    
    Args:
        text: Текст сегмента
        language: Язык текста
        
    Returns:
        Количество отрицаний
    """
    if language == 'ru':
        negations = ['не', 'нет', 'ни', 'никогда', 'ничего', 'никто']
    else:
        negations = ['not', 'no', 'never', 'nothing', 'nobody', 'none']
    
    words = text.lower().split()
    return sum(1 for word in words if word in negations)


def add_text_features(df: pd.DataFrame, language: str = 'ru') -> pd.DataFrame:
    """
    Добавляет текстовые признаки к датафрейму.
    
    Args:
        df: DataFrame с базовыми признаками
        language: Язык текста
        
    Returns:
        DataFrame с добавленными признаками
    """
    df = df.copy()
    
    df['speech_rate'] = df.apply(
        lambda row: calculate_speech_rate(
            str(row.get('text', '')),
            float(row.get('duration', 0))
        ),
        axis=1
    )
    
    df['articulation_rate'] = df.apply(
        lambda row: calculate_articulation_rate(
            str(row.get('text', '')),
            float(row.get('duration', 0)),
            float(row.get('pause_ratio', 0))
        ),
        axis=1
    )
    
    df['type_token_ratio'] = df['text'].apply(calculate_type_token_ratio)
    df['pronoun_count'] = df['text'].apply(lambda x: count_pronouns(str(x), language))
    df['negation_count'] = df['text'].apply(lambda x: count_negations(str(x), language))
    
    df['avg_word_length'] = df['text'].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0.0
    )
    
    return df


def merge_features(
    segments_metadata_path: str,
    opensmile_features_path: str,
    metadata_path: str = None,
    output_path: str = 'data/features/merged_features.csv',
    language: str = 'ru'
) -> pd.DataFrame:
    """
    Объединяет все признаки в единый датасет.
    
    Args:
        segments_metadata_path: Путь к CSV с метаданными сегментов
        opensmile_features_path: Путь к CSV с признаками openSMILE
        metadata_path: Путь к CSV с метаданными видео (id, label)
        output_path: Путь для сохранения объединенного датасета
        language: Язык текста
        
    Returns:
        Объединенный DataFrame
    """
    segments_df = pd.read_csv(segments_metadata_path)
    opensmile_df = pd.read_csv(opensmile_features_path)
    
    merged_df = segments_df.merge(
        opensmile_df,
        on=['segment_id', 'file_id'],
        how='inner'
    )
    
    if os.path.exists(output_path):
        existing_merged = pd.read_csv(output_path)
        existing_file_ids = set(existing_merged['file_id'].unique())
        new_file_ids = set(merged_df['file_id'].unique())
        
        if existing_file_ids & new_file_ids:
            print(f"Предупреждение: файлы {existing_file_ids & new_file_ids} уже есть в объединенном датасете. Перезаписываю.")
            merged_df = pd.concat([existing_merged[~existing_merged['file_id'].isin(new_file_ids)], merged_df], ignore_index=True)
        else:
            merged_df = pd.concat([existing_merged, merged_df], ignore_index=True)
            print(f"Добавлено {len(new_file_ids)} новых видео к существующим {len(existing_file_ids)}")
    
    merged_df = add_text_features(merged_df, language=language)
    
    if metadata_path and os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        if 'id' in metadata_df.columns and 'file_id' not in metadata_df.columns:
            metadata_df = metadata_df.rename(columns={'id': 'file_id'})
        
        if 'file_id' in metadata_df.columns:
            output_metadata_path = Path(output_path).parent / 'metadata_all.csv'
            
            if output_metadata_path.exists():
                existing_metadata = pd.read_csv(output_metadata_path)
                if 'id' in existing_metadata.columns and 'file_id' not in existing_metadata.columns:
                    existing_metadata = existing_metadata.rename(columns={'id': 'file_id'})
                
                combined_metadata = pd.concat([existing_metadata, metadata_df], ignore_index=True)
                combined_metadata = combined_metadata.drop_duplicates(subset=['file_id'], keep='last')
                combined_metadata.to_csv(output_metadata_path, index=False, encoding='utf-8')
                metadata_df = combined_metadata
                print(f"Метаданные объединены. Всего видео в metadata: {len(metadata_df)}")
            else:
                metadata_df.to_csv(output_metadata_path, index=False, encoding='utf-8')
                print(f"Метаданные сохранены в {output_metadata_path}")
            
            if 'label' in merged_df.columns:
                merged_df = merged_df.drop(columns=['label'])
            
            merged_df = merged_df.merge(
                metadata_df,
                on='file_id',
                how='left',
                suffixes=('', '_meta')
            )
            
            if 'label_meta' in merged_df.columns:
                merged_df['label'] = merged_df['label_meta']
                merged_df = merged_df.drop(columns=['label_meta'])
            elif 'label' not in merged_df.columns and 'label' in metadata_df.columns:
                merged_df['label'] = merged_df.merge(metadata_df, on='file_id', how='left')['label']
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Объединенный датасет сохранен в {output_path}")
    print(f"Всего строк: {len(merged_df)}")
    print(f"Всего признаков: {len(merged_df.columns)}")
    print(f"Уникальных видео: {merged_df['file_id'].nunique()}")
    
    return merged_df


def aggregate_per_video(
    merged_features_path: str,
    output_path: str = 'data/features/features_per_video.csv'
) -> pd.DataFrame:
    """
    Агрегирует признаки по видео (средние, стандартные отклонения и т.д.).
    
    Args:
        merged_features_path: Путь к объединенному датасету
        output_path: Путь для сохранения агрегированного датасета
        
    Returns:
        DataFrame с агрегированными признаками
    """
    df = pd.read_csv(merged_features_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['segment_id', 'start', 'end', 'duration']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = ['mean', 'std', 'min', 'max', 'median']
    
    grouped = df.groupby('file_id')[numeric_cols].agg(agg_dict)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    
    if 'label' in df.columns:
        labels = df.groupby('file_id')['label'].first()
        grouped['label'] = labels
    
    grouped = grouped.reset_index()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Агрегированный датасет сохранен в {output_path}")
    print(f"Всего видео: {len(grouped)}")
    
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description='Объединение аудио и текстовых признаков'
    )
    parser.add_argument(
        '--segments-metadata',
        type=str,
        default='data/segments/segments_metadata.csv',
        help='Путь к метаданным сегментов'
    )
    parser.add_argument(
        '--opensmile-features',
        type=str,
        default='data/features/opensmile_features.csv',
        help='Путь к признакам openSMILE'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='Путь к метаданным видео (id, label)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/features/merged_features.csv',
        help='Путь для сохранения объединенного датасета'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='ru',
        help='Язык текста'
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='Создать агрегированный датасет по видео'
    )
    
    args = parser.parse_args()
    
    merged_df = merge_features(
        args.segments_metadata,
        args.opensmile_features,
        args.metadata,
        args.output,
        args.language
    )
    
    if args.aggregate:
        aggregate_per_video(args.output)


if __name__ == '__main__':
    main()

