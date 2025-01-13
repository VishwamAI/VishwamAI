from setuptools import setup, find_packages

setup(
    name="vishwamai",
    version="0.1.0",
    packages=find_packages(include=['vishwamai', 'vishwamai.*']),
    install_requires=[
        "torch>=2.4.1",
        "torchvision>=0.15.1",
        "triton>=3.0.0",
        "safetensors>=0.4.5",
        "transformers>=4.46.3",
        "datasets>=3.2.0",
        "wandb>=0.19.2",
        "numpy==1.21.6",
        "tqdm>=4.67.1",
        "sentencepiece>=0.2.0",
    ],
    python_requires='>=3.8',
)
