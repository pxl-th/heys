from setuptools import (
    find_packages,
    setup,
)

VERSION = "0.0.1"

setup(
    name="hayes",
    version=VERSION,
    license="MIT",
    description="Implementation of Linear Cryptanalysis on Hayes block cipher",
    classifiers=[
        "Development Status :: 3 - Pre-Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    command_options={
        "build_sphinx": {
            "version": ("setup.py", VERSION),
            "release": ("setup.py", VERSION),
        },
    },
    packages=find_packages(),
    python_requires=">=3.7",
    setup_requires=[
        "pip>=19.1",
        "setuptools>=41.0",
        "pytest-runner>=4.4,<5",
    ],
    install_requires=[
        "numpy>=1.16",
    ],
    extras_require={
        "dev": [
            "sphinx>=2.0,<3.0",
            "sphinx-rtd-theme>=0.4",
        ],
    },
)