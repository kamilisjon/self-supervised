#!/usr/bin/env python3
from setuptools import setup

setup(name='self-supervised',
      version='1.0.0',
      install_requires=[
        "tinygrad==0.10.3",
        "opencv-python==4.11.0.86",
        "albumentations==2.0.7",
        "matplotlib==3.10.3",
        "numpy==2.2.5"
      ],
      python_requires='>=3.10',
      include_package_data=True)
