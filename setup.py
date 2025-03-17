from setuptools import setup, find_packages

setup(
    name="vishwamai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jax>=0.5.2",
        "jaxlib>=0.5.1",
        "flax>=0.10.4",
        "optax>=0.2.4",
        "transformers>=4.36.0",
        "torch>=2.6.0",
        "numpy>=1.26.4",
        "safetensors>=0.5.3",
        "sentencepiece>=0.2.0",
        "tokenizers>=0.15.0",
        "huggingface-hub>=0.29.2",
        "wandb>=1.39.2",
        "duckdb>=1.2.1",
        "tqdm>=4.67.1",
        "pyarrow>=16.1.0",
        "einops>=0.8.1",
        "chex>=0.1.89",
        "jaxtyping>=0.2.38",
        "optree>=0.14.1",
        "orbax-checkpoint>=0.11.8",
        "scipy>=1.11.4",
        "ml_collections>=1.0.0",
        "typing_extensions>=4.12.2"
    ],
    extras_require={
        "dev": [
            "ipython>=8.20.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.12"
        ],
        "tpu": [
            "cloud-tpu-client>=0.10",
            "libtpu-nightly"
        ]
    },
    description="TPU-optimized text-to-text generation model with knowledge distillation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Kasinadh Sarma",
    author_email="kasinadhsarma@gmail.com",
    url="https://github.com/kasinadhsarma/VishwamAI",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
