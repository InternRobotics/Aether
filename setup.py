#!/usr/bin/env python
import pathlib
import sys

import pkg_resources
from setuptools import find_packages, setup


PKG_NAME = "aether"
VERSION = "1.0.0"
EXTRAS = {}


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


def _fill_extras(extras):
    if extras:
        extras["all"] = list({item for group in extras.values() for item in group})
    return extras


version_range_max = max(sys.version_info[1], 10) + 1
setup(
    name=PKG_NAME,
    version=VERSION,
    author="Aether Team",
    author_email="tonghe90@gmail.com",
    url="https://github.com/OpenRobotLab/Aether",
    description="",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "World Model",
        "3D Vision",
        "Reconstruction",
        "Sythetic Data",
        "Embodied AI",
    ],
    license="MIT License",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ]
    + [f"Programming Language :: Python :: 3.{i}" for i in range(8, version_range_max)],
)
