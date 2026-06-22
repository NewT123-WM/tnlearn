from setuptools import setup, find_packages

setup(
    name='tnlearn-gpu',
    version='0.1.0',
    description='GPU-first wrappers for tnlearn (independent _gpu path)',
    packages=find_packages(include=['tnlearn_gpu', 'tnlearn_gpu.*']),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scikit-learn',
        'torch>=2.1.0',
        'torchinfo',
        'matplotlib',
        'tqdm',
    ],
)
