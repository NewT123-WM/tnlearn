import os
import re
from setuptools import setup, find_packages


def read_version():
    version_file = os.path.join("tnlearn", "__init__.py")
    with open(version_file, encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^__version__ = ['\"]([^'\"]*)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Unable to find version string.")

VERSION = read_version()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    name="tnlearn",
    version=VERSION,
    author="Meng WANG",
    author_email="wangmeng22@stu.hit.edu.cn",
    description="A Python package that uses task-based neurons to build neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NewT123-WM/tnlearn",  
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='~=3.8',
    
    install_requires=[
    'torch~=2.1.0',             
    'numpy~=1.26.0',
    'scikit-learn~=1.4.0',
    'pandas~=2.2.0',
    'sympy~=1.12',
    'tqdm~=4.66.0',
    'matplotlib~=3.8.0',
    'ipython~=8.18.0',
    'torchinfo~=1.8.0',
    'h5py~=3.10.0',
    'setuptools>=40.0.0'  
]
)
