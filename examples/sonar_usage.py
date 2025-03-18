"""Example usage of SONAR multilingual and multimodal capabilities in VishwamAI."""

from vishwamai.multimodal import load_multimodal_pipeline
from vishwamai.multimodal.config import create_default_multimodal_config
import numpy as np

def main():
    # Create configuration with SONAR enabled
    config = create_default_multimodal_config(include_sonar=True)
    
    # Initialize multilingual pipeline
    pipeline = load_multimodal_pipeline(
        "vishwamai/base",  # Replace with your model path
        task="multilingual",
        config=config
    )
    
    # Example 1: Text Translation
    text = "Hello, how are you?"
    translated = pipeline.translate(
        text=text,
        src_lang="eng",  # English
        tgt_lang="fra"   # French
    )
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    
    # Example 2: Cross-lingual Similarity
    texts1 = ["Hello world", "Good morning"]
    texts2 = ["Bonjour le monde", "Bon matin"]
    
    similarity = pipeline.compute_similarity(
        texts1=texts1,
        texts2=texts2,
        lang1="eng",  # English
        lang2="fra"   # French
    )
    print("\nCross-lingual similarity scores:")
    for t1, t2, score in zip(texts1, texts2, similarity):
        print(f"{t1} <-> {t2}: {score:.3f}")
    
    # Example 3: Batch Translation
    texts = [
        "The weather is nice today",
        "I love programming",
        "Machine learning is fascinating"
    ]
    
    translations = pipeline.batch_translate(
        texts=texts,
        src_lang="eng",  # English
        tgt_lang="deu",  # German
        batch_size=2
    )
    
    print("\nBatch translations:")
    for src, tgt in zip(texts, translations):
        print(f"{src} -> {tgt}")
    
    # Example 4: Speech Processing
    # Note: Replace with actual audio data
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Get speech embeddings
    speech_emb = pipeline.embed(
        speech=audio,
        src_lang="eng",
        sampling_rate=sample_rate
    )
    print("\nSpeech embedding shape:", speech_emb["embeddings"].shape)

if __name__ == "__main__":
    main()