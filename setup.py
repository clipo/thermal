#!/usr/bin/env python3
"""
Setup script for SGD Toolkit

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="sgd-toolkit",
    version="1.0.0",
    author="Thermal SGD Detection Team",
    author_email="",
    description="Submarine Groundwater Discharge Detection from Thermal Imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sgd-toolkit",  # Update with actual URL
    packages=find_packages(include=['sgd_toolkit', 'sgd_toolkit.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'sgd-autodetect=scripts.sgd_autodetect:main',
            'sgd-viewer=scripts.sgd_viewer:main',
            'sgd-train=scripts.train_segmentation:main',
            'sgd-coverage=scripts.generate_coverage_map:main',
        ],
    },
    include_package_data=True,
    package_data={
        'sgd_toolkit': ['*.pkl', '*.json'],
    },
    zip_safe=False,
)
