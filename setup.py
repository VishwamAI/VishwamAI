"""
Setup file for Vishwamai package
"""
from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="vishwamai",
    version="0.1.0",
    author="Kasinadh Sarma",
    description="T4-optimized machine learning model with tree-based planning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kasinadhsarma/VishwamAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=22.0',
            'isort>=5.0',
            'mypy>=0.9',
            'flake8>=3.9'
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
            'myst-parser>=0.15'
        ]
    },
    entry_points={
        'console_scripts': [
            'vishwamai-train=vishwamai.training.scripts.train:main',
            'vishwamai-eval=vishwamai.training.scripts.evaluate:main'
        ]
    },
    package_data={
        'vishwamai': [
            'config/*.json',
            'data/tokenizer/*.model',
            'data/tokenizer/*.vocab'
        ]
    }
)
