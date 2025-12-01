import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_segments_metadata(segments_metadata_path: str) -> pd.DataFrame:
    if not os.path.exists(segments_metadata_path):
        print(f"warning: segments metadata not found: {segments_metadata_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(segments_metadata_path)
    
    if df.empty:
        print(f"warning: segments metadata is empty")
        return pd.DataFrame()
    
    required_cols = ['file_id', 'segment_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"missing required columns in segments metadata: {missing_cols}")
    
    return df


def load_opensmile_features(opensmile_features_path: str = None) -> pd.DataFrame:
    if opensmile_features_path is None:
        return pd.DataFrame()
    
    if not os.path.exists(opensmile_features_path):
        print(f"warning: opensmile features not found: {opensmile_features_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(opensmile_features_path)
    
    if 'segment_id' not in df.columns:
        raise ValueError("segment_id column missing in opensmile features")
    
    return df


def load_metadata(metadata_path: str) -> pd.DataFrame:
    if not os.path.exists(metadata_path):
        print(f"warning: metadata not found: {metadata_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(metadata_path)
    
    if 'id' not in df.columns:
        raise ValueError("id column missing in metadata")
    
    if 'label' not in df.columns:
        print("warning: label column missing in metadata")
    
    return df


def extract_text_features(text: str, language: str = 'ru') -> dict:
    features = {
        'text_length': len(text),
        'text_length_chars': len(text.replace(' ', '')),
        'word_count': len(text.split()) if text.strip() else 0,
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0.0,
        'sentence_count': text.count('.') + text.count('!') + text.count('?'),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'comma_count': text.count(','),
    }
    
    return features


def merge_features(
    segments_df: pd.DataFrame,
    opensmile_df: pd.DataFrame = None,
    metadata_df: pd.DataFrame = None,
    language: str = 'ru',
    aggregate: bool = False
) -> pd.DataFrame:
    if segments_df.empty:
        return pd.DataFrame()
    
    merged_df = segments_df.copy()
    
    if not opensmile_df.empty:
        merged_df = merged_df.merge(
            opensmile_df,
            on='segment_id',
            how='left'
        )
        print(f"merged opensmile features: {len(opensmile_df.columns)} columns")
    else:
        print("skipping opensmile features merge (not available)")
    
    if not metadata_df.empty:
        if 'label' in metadata_df.columns:
            merged_df = merged_df.merge(
                metadata_df[['id', 'label']],
                left_on='file_id',
                right_on='id',
                how='left'
            )
            merged_df = merged_df.drop(columns=['id'], errors='ignore')
            print(f"merged metadata: {merged_df['label'].notna().sum()} segments with labels")
        else:
            print("warning: label column not found in metadata")
    
    if 'text' in merged_df.columns:
        text_features = merged_df['text'].apply(
            lambda x: extract_text_features(str(x) if pd.notna(x) else '', language)
        )
        text_features_df = pd.DataFrame(text_features.tolist(), index=merged_df.index)
        merged_df = pd.concat([merged_df, text_features_df], axis=1)
        print(f"extracted text features: {len(text_features_df.columns)} columns")
    
    if aggregate:
        print("aggregating features by file_id...")
        
        exclude_cols = ['file_id', 'segment_id', 'text', 'segment_path', 'label']
        feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
        
        numeric_cols = merged_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = ['mean', 'std', 'min', 'max']
        
        if 'text' in merged_df.columns:
            agg_dict['text'] = ' '.join
        
        if 'label' in merged_df.columns:
            agg_dict['label'] = 'first'
        
        aggregated = merged_df.groupby('file_id').agg(agg_dict).reset_index()
        
        aggregated.columns = [
            '_'.join(col).strip('_') if col[1] else col[0]
            for col in aggregated.columns.values
        ]
        
        if 'label_first' in aggregated.columns:
            aggregated = aggregated.rename(columns={'label_first': 'label'})
        
        merged_df = aggregated
        print(f"aggregated to {len(merged_df)} files")
    
    return merged_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--segments-metadata',
        type=str,
        required=True,
        help='segments metadata csv path'
    )
    parser.add_argument(
        '--opensmile-features',
        type=str,
        default=None,
        help='opensmile features csv path (optional)'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=None,
        help='metadata csv path with labels (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='output merged features csv path'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='ru',
        help='language for text feature extraction'
    )
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='aggregate features by file_id'
    )
    
    args = parser.parse_args()
    
    print("loading segments metadata...")
    segments_df = load_segments_metadata(args.segments_metadata)
    
    if segments_df.empty:
        print("no segments metadata found. skipping merge.")
        print("hint: run audio extraction, transcription, and segmentation first")
        return
    
    print(f"loaded {len(segments_df)} segments from {segments_df['file_id'].nunique()} files")
    
    opensmile_df = pd.DataFrame()
    if args.opensmile_features:
        print("loading opensmile features...")
        opensmile_df = load_opensmile_features(args.opensmile_features)
        if not opensmile_df.empty:
            print(f"loaded {len(opensmile_df)} opensmile feature rows")
    
    metadata_df = pd.DataFrame()
    if args.metadata:
        print("loading metadata...")
        metadata_df = load_metadata(args.metadata)
        if not metadata_df.empty:
            print(f"loaded {len(metadata_df)} metadata rows")
    
    print("merging features...")
    merged_df = merge_features(
        segments_df,
        opensmile_df=opensmile_df,
        metadata_df=metadata_df,
        language=args.language,
        aggregate=args.aggregate
    )
    
    if merged_df.empty:
        print("warning: merged dataframe is empty")
        return
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged_df.to_csv(args.output, index=False, encoding='utf-8')
    
    print(f"\nmerged features saved to {args.output}")
    print(f"total rows: {len(merged_df)}")
    print(f"total columns: {len(merged_df.columns)}")
    
    if 'label' in merged_df.columns:
        label_counts = merged_df['label'].value_counts()
        print(f"label distribution:\n{label_counts}")
    
    if args.aggregate:
        print(f"unique files: {merged_df['file_id'].nunique()}")
    else:
        print(f"unique files: {merged_df['file_id'].nunique()}")
        print(f"unique segments: {merged_df['segment_id'].nunique()}")


if __name__ == '__main__':
    main()

