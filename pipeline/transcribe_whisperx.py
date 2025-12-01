import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import whisperx


def patch_torch_load():
    try:
        from omegaconf import ListConfig
        from omegaconf.base import ContainerMetadata
        import typing
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([ListConfig, ContainerMetadata, typing.Any])
    except ImportError:
        # omegaconf is optional; skip if not available
        pass
    
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        # Force weights_only=False for compatibility with older checkpoints
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load


patch_torch_load()


def transcribe_audio_file(
    audio_path: str,
    model,
    align_model,
    metadata,
    language: str = 'ru',
    device: str = 'cpu',
    batch_size: int = 16
) -> dict:
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, language=language, batch_size=batch_size)
    
    result = whisperx.align(
        result['segments'],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False
    )
    
    return result


def batch_transcribe(
    input_dir: str,
    output_dir: str,
    model_name: str = 'medium',
    language: str = 'ru',
    device: str = 'cpu',
    mode: str = 'lightweight'
) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(input_path.rglob('*.wav'))
    
    if not audio_files:
        print(f"no audio files found in {input_dir}")
        return
    
    print(f"found {len(audio_files)} audio files")
    print(f"loading whisper model: {model_name}")
    
    batch_size = 4 if mode == 'lightweight' else 16
    
    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type="int8" if device == 'cpu' else "float16"
    )
    
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device
    )
    
    success_count = 0
    for audio_file in tqdm(audio_files, desc="transcribing"):
        relative_path = audio_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            print(f"skipping {audio_file.name}: transcript already exists")
            continue
        
        try:
            result = transcribe_audio_file(
                str(audio_file),
                model,
                align_model,
                metadata,
                language=language,
                device=device,
                batch_size=batch_size
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            success_count += 1
        except Exception as e:
            print(f"error processing {audio_file.name}: {e}")
    
    print(f"processed: {success_count}/{len(audio_files)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/audio_wav',
        help='wav files directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/transcripts',
        help='json transcripts output directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='medium',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='whisper model size'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='ru',
        help='transcription language code'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='device for inference'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='lightweight',
        choices=['lightweight', 'server'],
        help='mode: lightweight or server'
    )
    
    args = parser.parse_args()
    
    batch_transcribe(
        args.input_dir,
        args.output_dir,
        model_name=args.model,
        language=args.language,
        device=args.device,
        mode=args.mode
    )


if __name__ == '__main__':
    main()

