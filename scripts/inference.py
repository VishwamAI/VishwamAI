#!/usr/bin/env python3
"""
Inference script for VishwamAI models.

Example usage:
    python inference.py --prompt "The future of AI is" --model ./checkpoints/model
    python inference.py --multimodal --text "Describe this image" --image path/to/image.jpg
"""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import time

from vishwamai import (
    ModelConfig, VishwamAIModel, TextGenerator, MultimodalGenerator,
    MultimodalProcessor, pipeline, get_hardware_info
)


def load_image(image_path: str) -> jnp.ndarray:
    """Load and preprocess image for model input."""
    try:
        from PIL import Image
        import numpy as np
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to standard size (224x224)
        image = image.resize((224, 224))
        
        # Convert to array and normalize
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        return jnp.array(image_array)[None, ...]  # [1, 224, 224, 3]
        
    except ImportError:
        raise ImportError("PIL required for image loading. Install with: pip install Pillow")


def load_audio(audio_path: str) -> jnp.ndarray:
    """Load and preprocess audio for model input."""
    try:
        import librosa
        
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec)
        
        # Add batch dimension
        return jnp.array(log_mel.T)[None, ...]  # [1, time, n_mels]
        
    except ImportError:
        raise ImportError("librosa required for audio loading. Install with: pip install librosa")


def main():
    parser = argparse.ArgumentParser(description='VishwamAI Inference')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to model config file')
    
    # Text generation arguments
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    
    # Multimodal arguments
    parser.add_argument('--multimodal', action='store_true', help='Enable multimodal inference')
    parser.add_argument('--text', type=str, help='Text input for multimodal inference')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    
    # Chat arguments
    parser.add_argument('--chat', action='store_true', help='Start interactive chat session')
    parser.add_argument('--system-prompt', type=str, 
                       default="You are a helpful AI assistant.", 
                       help='System prompt for chat')
    
    # Performance arguments
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("🤖 VishwamAI Inference")
    print("=" * 40)
    
    # Print hardware info
    hardware_info = get_hardware_info()
    print(f"Hardware: {hardware_info['num_devices']} devices ({', '.join(set(hardware_info['device_types']))})")
    
    # Load model configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            import json
            config_dict = json.load(f)
            model_config = ModelConfig(**config_dict.get('model_config', config_dict))
    else:
        # Default configuration
        model_config = ModelConfig(
            dim=2048,
            depth=24,
            heads=32,
            vocab_size=50304,
            max_seq_len=2048,
            enable_multimodal=args.multimodal
        )
    
    print(f"Model: {model_config.depth} layers, {model_config.dim} dim, {model_config.heads} heads")
    
    # Create pipeline
    if args.multimodal:
        generator = pipeline("multimodal-generation", model=args.model)
        print("🎭 Multimodal mode enabled")
    else:
        generator = pipeline("text-generation", model=args.model)
        print("📝 Text generation mode")
    
    # Benchmark mode
    if args.benchmark:
        print("\n⚡ Running performance benchmark...")
        
        # Create dummy input
        dummy_input = jnp.ones((1, 512), dtype=jnp.int32)
        
        # Warmup
        for _ in range(3):
            if hasattr(generator, 'model'):
                _ = generator.model.apply(generator.params, dummy_input, training=False)
        
        # Benchmark
        times = []
        for i in range(10):
            start_time = time.time()
            if hasattr(generator, 'model'):
                output = generator.model.apply(generator.params, dummy_input, training=False)
                output.block_until_ready()
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"  Run {i+1}: {(end_time - start_time)*1000:.2f}ms")
        
        avg_time = sum(times) / len(times)
        print(f"📊 Average inference time: {avg_time*1000:.2f}ms")
        print(f"📊 Throughput: {1/avg_time:.2f} samples/sec")
        return
    
    # Chat mode
    if args.chat:
        print("\n💬 Starting interactive chat session (type 'quit' to exit)")
        print(f"System: {args.system_prompt}")
        print("-" * 40)
        
        messages = [{"role": "system", "content": args.system_prompt}]
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    break
                
                messages.append({"role": "user", "content": user_input})
                
                # Generate response
                if hasattr(generator, 'chat'):
                    response = generator.chat(
                        messages,
                        max_length=args.max_length,
                        temperature=args.temperature
                    )
                else:
                    # Fallback to simple generation
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                    prompt += "\nassistant: "
                    response = generator.generate(
                        prompt,
                        max_length=args.max_length,
                        temperature=args.temperature
                    )
                
                print(f"Assistant: {response}")
                messages.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n👋 Chat session ended")
        return
    
    # Multimodal inference
    if args.multimodal:
        print("\n🎭 Multimodal inference")
        
        # Load media files
        images = None
        audio = None
        
        if args.image:
            print(f"📷 Loading image: {args.image}")
            images = load_image(args.image)
            print(f"Image shape: {images.shape}")
        
        if args.audio:
            print(f"🎵 Loading audio: {args.audio}")
            audio = load_audio(args.audio)
            print(f"Audio shape: {audio.shape}")
        
        # Generate response
        if hasattr(generator, 'generate_from_multimodal'):
            response = generator.generate_from_multimodal(
                text=args.text,
                images=images,
                audio=audio,
                max_length=args.max_length,
                temperature=args.temperature,
                seed=args.seed
            )
        else:
            response = "Multimodal generation not implemented in this model"
        
        print(f"\n🤖 Response: {response}")
    
    # Simple text generation
    elif args.prompt:
        print(f"\n📝 Generating from prompt: '{args.prompt}'")
        
        start_time = time.time()
        response = generator.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed
        )
        end_time = time.time()
        
        print(f"\n🤖 Response: {response}")
        print(f"⏱️  Generation time: {(end_time - start_time)*1000:.2f}ms")
    
    else:
        print("❌ Please provide --prompt, --chat, --multimodal, or --benchmark option")
        parser.print_help()


if __name__ == '__main__':
    main()
