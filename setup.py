from setuptools import setup, find_packages

setup(
    name="amt_hypervector",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.13",
        "numpy",
        "transformers",
        "python-chess",
        "pytest",
        "torchhd @ git+https://github.com/hyperdimensional-computing/torchhd.git@main#egg=torchhd",
    ],
    entry_points={
        "console_scripts": [
            "chess-demo=plugins.chess_toy.selfplay_chess:main",
        ],
    },
)
