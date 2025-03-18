from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="vishwamai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest==8.3.5",
            "pytest-cov==6.0.0",
            "black==24.1.1",
            "isort==5.13.0",
            "flake8==7.1.2",
            "mypy==1.6.0",
            "pytest-xdist==3.3.0"
        ],
        "tpu": [
            "cloud-tpu-client==0.10",
            "libtpu-nightly"
        ],
        "fairseq2": [
            "fairseq2"
        ]
    },
    description="Advanced language model training system with Hydra configuration, hyperparameter tuning, and distributed training support",
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8, <4.0",
)
