#!/usr/bin/env python3
"""
Setup script for the Rapa Nui SGD Toolkit.

Editable install (recommended for development):
    pip install -e .

Standard install:
    pip install .

The toolkit is exposed as the importable package `sgd_toolkit`. Operational
scripts live under `scripts/<subdir>/X.py` and are invoked directly with
`python scripts/<subdir>/X.py [args]` — see scripts/README.md.
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent

readme = ROOT / "README.md"
long_description = readme.read_text() if readme.exists() else ""

# Parse requirements.txt — strip comments, blank lines, and option lines.
requirements_file = ROOT / "requirements.txt"
requirements: list[str] = []
if requirements_file.exists():
    for raw in requirements_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # Drop trailing inline comments ("pkg>=1.0  # note")
        line = line.split("#", 1)[0].strip()
        if line:
            requirements.append(line)

setup(
    name="sgd-toolkit",
    version="1.1.0",
    author="Carl Lipo",
    author_email="clipo@binghamton.edu",
    description=(
        "Quantitative Submarine Groundwater Discharge (SGD) detection from "
        "thermal-drone surveys — Rapa Nui pipeline."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clipo/thermal",
    packages=find_packages(include=["sgd_toolkit", "sgd_toolkit.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "paper": [
            "python-docx>=0.8.11",
        ],
        "sam": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            # segment-anything is git-only — install via scripts/setup_sam.sh
        ],
    },
    include_package_data=True,
    package_data={
        "sgd_toolkit": ["*.pkl", "*.json"],
    },
    zip_safe=False,
)
