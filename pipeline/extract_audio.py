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
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-ac', '1',
            '-ar', str(sample_rate),
            '-af', f'loudnorm=I={normalize_db}:TP=-1.5:LRA=11',
            '-y',
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
        print(f"error processing {video_path}: {e.stderr}")
        return False
    except Exception as e:
        print(f"unexpected error: {e}")
        return False


def batch_extract_audio(
    input_dir: str,
    output_dir: str,
    video_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.rglob(f'*{ext}'))
    
    if not video_files:
        print(f"no video files found in {input_dir}")
        return
    
    print(f"found {len(video_files)} video files")
    
    success_count = 0
    for video_file in tqdm(video_files, desc="extracting audio"):
        relative_path = video_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.wav')
        
        if extract_audio_from_video(str(video_file), str(output_file)):
            success_count += 1
        else:
            print(f"skipped file: {video_file.name}")
    
    print(f"processed: {success_count}/{len(video_files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw_videos',
        help='video files directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/audio_wav',
        help='wav output directory'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='sample rate (default 16000)'
    )
    
    args = parser.parse_args()
    
    batch_extract_audio(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

