from setuptools import setup, find_packages

setup(
    name="vishwamai",
    version="0.1.0",
    author="Kasinadh Sarma",
    author_email="kasinadhsarma@gmail.com",
    description="A math-focused machine learning library with efficient quantization and advanced tokenization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kasinadhsarma/VishwamAI",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.13.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "pyarrow>=14.0.1",
        "triton>=2.1.0",
        "pytest>=8.0.0",
        "sentencepiece>=0.2.0",
        "tqdm>=4.65.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
