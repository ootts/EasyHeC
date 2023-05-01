from setuptools import find_packages
from setuptools import setup
setup(
    name="easyhec",
    version="0.1",
    author="Linghao Chen",
    packages=find_packages(exclude=("configs", "tests", "models", "data", "dbg")),
)
