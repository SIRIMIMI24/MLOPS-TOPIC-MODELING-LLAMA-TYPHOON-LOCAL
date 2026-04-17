from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOPS-TOPIC-MODELING-LLAMA-TYPHOON-LOCAL",
    version="0.1",
    author="SIRIDACH JAROENSIRI",
    packages=find_packages(),
    install_requires = requirements,
)