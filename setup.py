from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tnlearn",
    version="0.1.0",
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
    python_requires='>=3.9',
    install_requires=[
        "h5py~=3.10.0",
        "numpy~=1.26.2",
        "torch~=2.1.0",
        "sympy~=1.12",
        "scikit-learn~=1.4.0",
        "scipy~=1.12.0",
        "joblib~=1.3.2",
        "requests~=2.31.0",
        "networkx~=3.2.1",
        "matplotlib~=3.8.3",
        "pandas~=2.2.0",
        "packaging~=23.2",
        "ipython~=8.18.1",
        "tqdm~=4.66.2",
    ],
)
