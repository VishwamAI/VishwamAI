from setuptools import setup, find_packages

setup(
    name="vishwamai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.1",
        "transformers>=4.46.3",
        "pyarrow>=3.0.0",
        "numpy>=1.21.6",
        "pandas>=1.3.0",
    ],
)
