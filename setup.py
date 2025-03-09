#!/usr/bin/env python

from setuptools import setup, find_packages
import pkg_resources  

def read_requirements(file_path="requirements.txt"):
    with open(file_path) as f:
        requirements = [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(f)
        ]
    return requirements

setup(
    name="moore_tsr",  #
    version="0.0.1",  
    description="Simple package to play with TSR and moore dataset", 
    author="Salif SAWADOGO",  
    url="https://github.com/sawadogosalif/SaChi",
    packages=find_packages("src"), 
    package_dir={"": "src"},  
    install_requires=read_requirements(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  
)