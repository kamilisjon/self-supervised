#!/usr/bin/env python3

from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
  long_description = f.read()

setup(name='self-supervised',
      version='1.0.0',
      author='Kamilis Jonkus',
      license='MIT',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=[
        "tinygrad==0.10.3",
        "opencv-python==4.11.0.86",
        "albumentations==2.0.7",
        "matplotlib==3.10.3"
      ],
      python_requires='>=3.10',
      include_package_data=True)