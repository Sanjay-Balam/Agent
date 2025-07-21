#!/usr/bin/env python3
"""
Setup script for the Enhanced Multi-Domain LLM project.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("docs/README_ENHANCED.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("src/config/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-manim-llm",
    version="2.0.0",
    author="Enhanced Manim Agent Team",
    description="A powerful AI assistant trained on Manim animations, Data Structures & Algorithms, and System Design concepts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "gpu": ["torch[cuda]"],
    },
    entry_points={
        "console_scripts": [
            "manim-llm-train=scripts.train:main",
            "manim-llm-api=scripts.api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
)