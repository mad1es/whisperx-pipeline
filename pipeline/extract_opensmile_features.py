import os
import subprocess
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


def find_smilextract_binary(opensmile_dir: str = "opensmile") -> str:
    possible_paths = [
        os.path.join(opensmile_dir, "build", "progsrc", "smilextract", "SMILExtract"),
        os.path.join(opensmile_dir, "build", "bin", "SMILExtract"),
        "SMILExtract"
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return os.path.abspath(path)
    
    raise FileNotFoundError(
        f"smilextract not found. check paths: {possible_paths}"
    )


def find_egemaps_config(opensmile_dir: str = "opensmile") -> str:
    config_path = os.path.join(
        opensmile_dir,
        "config",
        "egemaps",
        "v02",
        "eGeMAPSv02.conf"
    )
    
    if os.path.exists(config_path):
        return os.path.abspath(config_path)
    
    raise FileNotFoundError(
        f"egemaps config file not found: {config_path}"
    )


def extract_features_opensmile(
    audio_path: str,
    output_path: str,
    smilextract_path: str,
    config_path: str
) -> bool:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            smilextract_path,
            '-C', config_path,
            '-I', audio_path,
            '-O', output_path,
            '-csvoutput', '1',
            '-l', '1',
            '-nologfile', '1'
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        return os.path.exists(output_path)
        
    except subprocess.CalledProcessError as e:
        print(f"error processing {audio_path}: {e.stderr}")
        return False
    except Exception as e:
        print(f"unexpected error: {e}")
        return False


def parse_opensmile_csv(csv_path: str) -> dict:
    try:
        if not os.path.exists(csv_path):
            return {}
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        if len(lines) < 2:
            return {}
        
        features = {}
        attribute_names = []
        data_started = False
        data_line = None
        
        for line in lines:
            line = line.strip()
            
            if not line or line.startswith('%'):
                continue
            
            if line.startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1]
                    if attr_name != 'name':
                        attribute_names.append(attr_name)
            
            elif line.startswith('@data'):
                data_started = True
                continue
            
            elif data_started and line:
                data_line = line.split(',')
                break
        
        if not data_line or len(attribute_names) == 0:
            return {}
        
        for i, attr_name in enumerate(attribute_names):
            data_idx = i + 1
            if data_idx < len(data_line):
                try:
                    value_str = data_line[data_idx].strip().strip('"\'')
                    if value_str and value_str != '?' and value_str != 'unknown':
                        value = float(value_str)
                        if not np.isnan(value) and not np.isinf(value):
                            features[attr_name] = value
                except (ValueError, IndexError):
                    continue
        
        return features
        
    except Exception as e:
        print(f"error parsing {csv_path}: {e}")
        return {}


def batch_extract_features(
    segments_dir: str,
    output_dir: str,
    opensmile_dir: str = "opensmile"
) -> pd.DataFrame:
    segments_path = Path(segments_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    smilextract_path = find_smilextract_binary(opensmile_dir)
    config_path = find_egemaps_config(opensmile_dir)
    
    print(f"using smilextract: {smilextract_path}")
    print(f"using config: {config_path}")
    
    all_features = []
    
    video_dirs = [d for d in segments_path.iterdir() if d.is_dir()]
    
    for video_dir in tqdm(video_dirs, desc="extracting features"):
        segment_files = list(video_dir.glob('segment_*.wav'))
        
        for segment_file in segment_files:
            segment_id = f"{video_dir.name}_{segment_file.stem}"
            output_csv = output_path / f"{segment_id}.csv"
            
            if extract_features_opensmile(
                str(segment_file),
                str(output_csv),
                smilextract_path,
                config_path
            ):
                features = parse_opensmile_csv(str(output_csv))
                if features:
                    features['segment_id'] = segment_id
                    features['file_id'] = video_dir.name
                    all_features.append(features)
    
    if all_features:
        features_df = pd.DataFrame(all_features)
        
        output_csv_path = Path(output_dir) / 'opensmile_features.csv'
        if output_csv_path.exists():
            existing_df = pd.read_csv(output_csv_path)
            existing_file_ids = set(existing_df['file_id'].unique())
            new_file_ids = set(features_df['file_id'].unique())
            
            if existing_file_ids & new_file_ids:
                print(f"warning: files {existing_file_ids & new_file_ids} already exist, overwriting")
                features_df = pd.concat([existing_df[~existing_df['file_id'].isin(new_file_ids)], features_df], ignore_index=True)
            else:
                features_df = pd.concat([existing_df, features_df], ignore_index=True)
                print(f"added {len(new_file_ids)} new videos to existing {len(existing_file_ids)}")
        
        return features_df
    else:
        output_csv_path = Path(output_dir) / 'opensmile_features.csv'
        if output_csv_path.exists():
            return pd.read_csv(output_csv_path)
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--segments-dir',
        type=str,
        default='data/segments',
        help='audio segments directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/features',
        help='features output directory'
    )
    parser.add_argument(
        '--opensmile-dir',
        type=str,
        default='opensmile',
        help='opensmile directory'
    )
    parser.add_argument(
        '--output-csv',
        type=str,
        default='data/features/opensmile_features.csv',
        help='output csv path'
    )
    
    args = parser.parse_args()
    
    features_df = batch_extract_features(
        args.segments_dir,
        args.output_dir,
        args.opensmile_dir
    )
    
    if not features_df.empty:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        features_df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"features saved to {args.output_csv}")
        print(f"total segments: {len(features_df)}")
        print(f"total features: {len(features_df.columns)}")
        print(f"unique videos: {features_df['file_id'].nunique()}")
    else:
        print("failed to extract features")


if __name__ == '__main__':
    main()

