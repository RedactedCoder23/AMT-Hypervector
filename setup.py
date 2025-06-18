from setuptools import setup, find_packages

setup(
    name='amt_hypervector',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.13',
        'transformers',
        'numpy',
        'python-chess',
        'torch-hd @ git+https://github.com/hyperdimensional-computing/torchhd.git'
    ],
)
