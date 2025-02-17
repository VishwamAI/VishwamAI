from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vishwamai",
    version="0.1.1",
    author="Vishwamai Contributors",
    author_email="your.email@example.com",
    description="Advanced AI Training Framework with Emergent Behavior and Ethical Considerations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/vishwamai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "transformers>=4.5.0",
        "dataclasses",
        "typing-extensions",
        "tqdm",
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black',
            'isort',
            'mypy',
            'pylint'
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme',
            'sphinx-autodoc-typehints'
        ],
        'examples': [
            'jupyter',
            'matplotlib',
            'pandas'
        ]
    },
    entry_points={
        'console_scripts': [
            'vishwamai-train=vishwamai.examples.basic_training:main',
        ],
    },
    package_data={
        'vishwamai': [
            'README.md',
            'LICENSE',
            'examples/*.py',
        ],
    },
    include_package_data=True,
)
