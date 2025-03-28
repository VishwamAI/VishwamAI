"""Setup for VishwamAI kernel tests."""

import os
from setuptools import setup, find_namespace_packages

def read_requirements(filename: str) -> list:
    """Read requirements from file."""
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read requirements
requirements = read_requirements('requirements-test.txt')

# Platform-specific requirements
extra_requires = {
    'tpu': [
        'jax[tpu]>=0.4.1',
        'jaxlib>=0.4.1',
        'libtpu-nightly'
    ],
    'gpu': [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'cuda-python>=11.8',
        'cupy-cuda11x>=12.0.0'
    ],
    'dev': [
        'black>=22.0.0',
        'flake8>=4.0.0',
        'mypy>=0.900',
        'isort>=5.10.0'
    ],
    'benchmark': [
        'pytest-benchmark>=4.0.0',
        'memory-profiler>=0.60.0',
        'scalene>=1.5.19'
    ]
}

# All extras combined
extra_requires['all'] = sorted({
    pkg for pkgs in extra_requires.values() for pkg in pkgs
})

setup(
    name='vishwamai-kernel-tests',
    version='0.1.0',
    description='Test suite for VishwamAI kernels',
    author='VishwamAI Team',
    packages=find_namespace_packages(include=['tests.*']),
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require=extra_requires,
    entry_points={
        'console_scripts': [
            'run-kernel-tests=tests.run_tests:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    test_suite='tests',
    include_package_data=True,
    zip_safe=False,
)
