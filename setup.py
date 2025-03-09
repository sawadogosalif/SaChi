#!/usr/bin/env python

"""The setup script."""

src_path = "src"

import os, sys, runpy
import pathlib

from setuptools import setup, find_packages
import pkg_resources

# get metadata
metadata_path = next(pathlib.Path(src_path).glob("*/package_metadata.py"))
metadata = runpy.run_path(metadata_path)

author = metadata["__author__"]
email = metadata["__email__"]
doc = metadata["__doc__"]
name = metadata["__name__"]
url = metadata["__url__"]
version = metadata["__version__"]

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

with open("requirements.txt") as requirements_file:
    requirements = [
        str(requirement)
        for requirement in pkg_resources.parse_requirements(requirements_file)
    ]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    packages=find_packages(src_path),
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.json files, include them:
        "": ["*.txt", "*.json"],
    },
    package_dir={"": src_path},
    name=name,
    author=author,
    author_email=email,
    python_requires=">=3.8",
    classifiers=[],
    description=doc,
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    keywords=name,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url=url,
    version=version,
    zip_safe=False,
)