"""Setup script for vishwamai package."""

import os
from setuptools import setup, find_packages

def read_requirements(filename: str) -> list:
    """Read requirements from file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def get_version() -> str:
    """Get package version."""
    init_path = os.path.join('vishwamai', '__init__.py')
    with open(init_path) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'").strip('"')
    return '0.1.0'

# Read package requirements
install_requires = [
    'torch>=2.0.0',
    'torch-xla>=2.0.0',
    'transformers>=4.30.0',
    'sentencepiece>=0.1.99',
    'numpy>=1.24.0',
    'pyyaml>=6.0',
    'tqdm>=4.65.0',
    'fastapi>=0.95.0',
    'uvicorn>=0.22.0',
    'tensorboard>=2.13.0',
    'wandb>=0.15.0',
    'einops>=0.6.1',
    'flash-attn>=2.0.0',
    'bitsandbytes>=0.41.0',
    'datasets>=2.13.0',
    'accelerate>=0.21.0',
    'fairscale>=0.4.13',
    'triton>=2.0.0',
    'huggingface-hub>=0.16.4',
    'scipy>=1.10.0',
    'matplotlib>=3.7.1',
    'jax',
    'jaxlib',
    'flax',
    'optax',
    'safetensors',
]

# Extra requirements for development
extras_require = {
    'dev': [
        'pytest>=7.3.1',
        'pytest-cov>=4.1.0',
        'black>=23.3.0',
        'isort>=5.12.0',
        'flake8>=6.0.0',
        'mypy>=1.3.0',
        'pylint>=2.17.4',
    ],
    'docs': [
        'sphinx>=6.2.1',
        'sphinx-rtd-theme>=1.2.2',
        'sphinx-autodoc-typehints>=1.23.0',
    ],
    'profiling': [
        'tensorboard-plugin-profile>=2.13.0',
        'py-spy>=0.3.14',
        'memory_profiler>=0.61.0',
    ],
}

setup(
    name='vishwamai',
    version=get_version(),
    description='VishwamAI language model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='KasinadhSarma',
    author_email='research@example.com',
    url='https://github.com/organization/vishwamai',
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'vishwamai-train=vishwamai.scripts.train_model:main',
            'vishwamai-eval=vishwamai.scripts.evaluate_model:main',
            'vishwamai-serve=vishwamai.scripts.serve_model:main',
            'vishwamai-preprocess=vishwamai.scripts.preprocess_data:main',
            'vishwamai-tokenizer=vishwamai.scripts.train_tokenizer:main',
            'vishwamai-export=vishwamai.scripts.export_model:main',
        ],
    },
    include_package_data=True,
    package_data={
        'vishwamai': [
            'configs/*.yaml',
            'data/**/*.json',
            'model/**/*.py',
        ],
    },
    zip_safe=False,
    keywords=[
        'deep-learning',
        'machine-learning',
        'natural-language-processing',
        'mixture-of-experts',
        'multi-level-attention',
        'transformer',
        'language-model',
        'tpu',
    ],
    project_urls={
        'Documentation': 'https://vishwamai.readthedocs.io/',
        'Source': 'https://github.com/organization/vishwamai',
        'Tracker': 'https://github.com/organization/vishwamai/issues',
    },
)
