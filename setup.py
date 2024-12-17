from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="NeuralSolvers",
    version="0.1.0",
    author="Nico Hoffmann, Patrick Stiller, Maksim Zhdanov, Jeyhun Rustamov, Raj Dhansukhbhai Sutariya",
    author_email="nico.hoffmann@saxony.ai",
    description="A framework for solving partial differential equations (PDEs) and inverse problems using physics-informed neural networks (PINNs) at scale",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Photon-AI-Research/NeuralSolvers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
    ],
    extras_require={
    },
    entry_points={
        "console_scripts": [
            "neuralsolvers=NeuralSolvers.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "NeuralSolvers": ["examples/*"],
    },
)