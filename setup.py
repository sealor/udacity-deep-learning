from setuptools import setup, find_packages

setup(
    name="udacity-deep-learning",
    author="Stefan Richter",
    packages=find_packages(exclude="test"),
    install_requires=[
        "numpy", "sklearn", "scipy", "matplotlib"
    ]
)
