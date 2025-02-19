from setuptools import setup, find_packages

setup(
    name="vishwamai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "safetensors>=0.3.0",
        "wandb>=0.15.0",
        "datasets>=2.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.16.0",
    ],
    author="Kasinadhsarma",
    author_email="contact@vishwamai.ai",
    description="VishwamAI Language Model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VishwamAI/VishwamAI",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
