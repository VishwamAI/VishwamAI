#!/usr/bin/env python3
"""
Setup script for VishwamAI.
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    """Read requirements from a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core requirements
install_requires = [
    'jax>=0.4.13',
    'jaxlib>=0.4.13', 
    'flax>=0.7.5',
    'optax>=0.1.7',
    'transformers>=4.36.0',
    'torch==2.7.1',
    'numpy>=1.24.0',
    'safetensors>=0.4.0',
    'datasets>=2.14.0',
    'sentencepiece==0.2.0',
    'tokenizers>=0.15.0',
    'huggingface-hub>=0.19.0',
    'einops==0.8.1',
    'chex==0.1.89',
    'jaxtyping==0.2.38',
    'optree==0.14.1',
    'orbax-checkpoint==0.11.8',
    'tqdm>=4.65.0',
    'scipy==1.11.4',
    'ml_collections==1.0.0',
    'typing_extensions==4.12.2',
]

# Development requirements
dev_requires = [
    'pytest>=7.0.0',
    'pytest-cov>=4.1.0',
    'pytest-xdist>=3.3.0',
    'pytest-env>=1.1.0',
    'ipython==8.20.0',
    'jupyter==1.0.0',
    'notebook==6.4.12',
    'matplotlib>=3.8.0',
    'psutil>=5.9.0',
]

# Optional requirements for different features
extras_require = {
    'dev': dev_requires,
    'wandb': ['wandb>=0.15.0'],
    'data': ['duckdb>=0.9.0', 'pandas>=2.0.0', 'pyarrow==16.1.0'],
    'hydra': ['hydra-core==1.3.2'],
    'sonar': [
        'fairseq2',
        'editdistance~=0.8',
        'importlib_metadata~=7.0', 
        'importlib_resources~=6.4',
        'sacrebleu~=2.4',
    ],
    'all': dev_requires + [
        'wandb>=0.15.0',
        'duckdb>=0.9.0',
        'pandas>=2.0.0',
        'pyarrow==16.1.0',
        'hydra-core==1.3.2',
        'fairseq2',
        'editdistance~=0.8',
        'importlib_metadata~=7.0',
        'importlib_resources~=6.4', 
        'sacrebleu~=2.4',
    ]
}

setup(
    name="vishwamai",
    version="0.1.0",
    author="VishwamAI Team",
    author_email="contact@vishwamai.ai",
    description="Efficient multimodal AI framework with curriculum learning support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/VishwamAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    # entry_points={
    #     "console_scripts": [
    #         "vishwamai-setup=setup_vishwamai:main",
    #     ],
    # },
    include_package_data=True,
    package_data={
        "vishwamai": ["configs/*.json"],
    },
    zip_safe=False,
    keywords="ai machine-learning deep-learning jax flax multimodal curriculum-learning",
    project_urls={
        "Bug Reports": "https://github.com/VishwamAI/VishwamAI/issues",
        "Documentation": "https://github.com/VishwamAI/VishwamAI/tree/main/docs",
        "Source": "https://github.com/VishwamAI/VishwamAI",
    },
)
