from setuptools import setup, find_packages

setup(
    name="federated-learning-framework",
    version="0.1.0",
    description="A minimal federated learning framework for experiments",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.62.0",
        "wandb>=0.15.0",
        "PyYAML>=6.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
