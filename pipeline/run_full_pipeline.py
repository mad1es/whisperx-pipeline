import os
import sys
import argparse
import subprocess
import platform
import psutil
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    print(f"running: {description}")
    print(f"command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False
        )
        print(f"{description} completed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"error executing {description}")
        print(f"exit code: {e.returncode}\n")
        return False
    except FileNotFoundError:
        print(f"command not found: {cmd[0]}")
        print("ensure all dependencies are installed\n")
        return False


def detect_environment():
    is_mac = platform.system() == 'Darwin'
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    if is_mac or total_memory_gb < 16:
        return {
            'mode': 'lightweight',
            'default_model': 'medium',
            'description': 'lightweight mode'
        }
    else:
        return {
            'mode': 'server',
            'default_model': 'medium',
            'description': 'server mode'
        }


def check_dependencies() -> bool:
    print("checking dependencies...")
    
    dependencies = {
        'python': ['python', '--version'],
        'ffmpeg': ['ffmpeg', '-version'],
    }
    
    all_ok = True
    for name, cmd in dependencies.items():
        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True
            )
            print(f"{name} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"{name} not found")
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--skip-audio',
        action='store_true',
        help='skip audio extraction'
    )
    parser.add_argument(
        '--skip-transcription',
        action='store_true',
        help='skip transcription'
    )
    parser.add_argument(
        '--skip-segmentation',
        action='store_true',
        help='skip segmentation'
    )
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='skip feature extraction'
    )
    parser.add_argument(
        '--skip-merge',
        action='store_true',
        help='skip feature merging'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='skip model training'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='data directory'
    )
    parser.add_argument(
        '--whisper-model',
        type=str,
        default=None,
        help='whisper model for transcription (none = auto-detect)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['lightweight', 'server'],
        help='mode: lightweight or server (none = auto-detect)'
    )
    parser.add_argument(
        '--whisper-language',
        type=str,
        default='ru',
        help='transcription language'
    )
    parser.add_argument(
        '--whisper-device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='whisper device'
    )
    
    args = parser.parse_args()
    
    env = detect_environment()
    if args.mode:
        env['mode'] = args.mode
        if args.mode == 'lightweight':
            env['default_model'] = 'medium'
            env['batch_size'] = 4
        else:
            env['default_model'] = 'medium'
            env['batch_size'] = 16
    
    if args.whisper_model is None:
        args.whisper_model = env['default_model']
    
    print(f"environment: {env['description']}")
    print(f"mode: {env['mode']}")
    print(f"whisper model: {args.whisper_model}")
    
    if not check_dependencies():
        print("some dependencies not found. continue? (y/n)")
        response = input().lower()
        if response != 'y':
            sys.exit(1)
    
    base_dir = Path(args.data_dir)
    pipeline_dir = Path('pipeline')
    
    success = True
    
    if not args.skip_audio:
        raw_videos_dir = base_dir / 'raw_videos'
        audio_wav_dir = base_dir / 'audio_wav'
        
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        videos_in_raw = list(raw_videos_dir.rglob('*')) if raw_videos_dir.exists() else []
        videos_in_raw = [f for f in videos_in_raw if f.suffix.lower() in video_extensions]
        
        videos_in_audio = list(audio_wav_dir.rglob('*')) if audio_wav_dir.exists() else []
        videos_in_audio = [f for f in videos_in_audio if f.suffix.lower() in video_extensions]
        
        if videos_in_raw:
            input_dir = str(raw_videos_dir)
        elif videos_in_audio:
            print(f"found videos in {audio_wav_dir}, using as input")
            input_dir = str(audio_wav_dir)
        else:
            input_dir = str(raw_videos_dir)
        
        success &= run_command(
            [
                'python', str(pipeline_dir / 'extract_audio.py'),
                '--input-dir', input_dir,
                '--output-dir', str(audio_wav_dir)
            ],
            'extracting audio from video'
        )
    
    if success and not args.skip_transcription:
        transcribe_cmd = [
            'python', str(pipeline_dir / 'transcribe_whisperx.py'),
            '--input-dir', str(base_dir / 'audio_wav'),
            '--output-dir', str(base_dir / 'transcripts'),
            '--model', args.whisper_model,
            '--language', args.whisper_language,
            '--device', args.whisper_device
        ]
        if args.mode:
            transcribe_cmd.extend(['--mode', args.mode])
        
        success &= run_command(transcribe_cmd, 'transcribing audio')
    
    if success and not args.skip_segmentation:
        success &= run_command(
            [
                'python', str(pipeline_dir / 'segment_audio.py'),
                '--audio-dir', str(base_dir / 'audio_wav'),
                '--transcript-dir', str(base_dir / 'transcripts'),
                '--output-dir', str(base_dir / 'segments')
            ],
            'segmenting audio'
        )
    
    if success and not args.skip_features:
        success &= run_command(
            [
                'python', str(pipeline_dir / 'extract_opensmile_features.py'),
                '--segments-dir', str(base_dir / 'segments'),
                '--output-dir', str(base_dir / 'features'),
                '--output-csv', str(base_dir / 'features' / 'opensmile_features.csv')
            ],
            'extracting opensmile features'
        )
    
    if success and not args.skip_merge:
        metadata_path = base_dir / 'metadata.csv'
        opensmile_features_path = base_dir / 'features' / 'opensmile_features.csv'
        
        merge_cmd = [
            'python', str(pipeline_dir / 'merge_features.py'),
            '--segments-metadata', str(base_dir / 'segments' / 'segments_metadata.csv'),
            '--output', str(base_dir / 'features' / 'merged_features.csv'),
            '--language', args.whisper_language,
            '--aggregate'
        ]
        
        if not args.skip_features and opensmile_features_path.exists():
            merge_cmd.extend(['--opensmile-features', str(opensmile_features_path)])
        
        if metadata_path.exists():
            merge_cmd.extend(['--metadata', str(metadata_path)])
        
        success &= run_command(
            merge_cmd,
            'merging features'
        )
    
    if success and not args.skip_training:
        merged_features = base_dir / 'features' / 'merged_features.csv'
        if merged_features.exists():
            success &= run_command(
                [
                    'python', str(pipeline_dir / 'train_baseline.py'),
                    '--data', str(merged_features),
                    '--model-type', 'logistic',
                    '--n-splits', '5'
                ],
                'training baseline model'
            )
        else:
            print(f"file {merged_features} not found, skipping training")
    
    if success:
        print("pipeline success")
        print("check results in data/features/merged_features.csv")
        print("run visualization: streamlit run visualization_app/app.py")
    else:
        print("pipeline failed")
        print("check logs above for details")
        sys.exit(1)


if __name__ == '__main__':
    main()

