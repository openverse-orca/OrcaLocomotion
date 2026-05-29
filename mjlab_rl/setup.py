"""Installation script for the vendored 'mjlab_rl' python package."""

from setuptools import setup, find_namespace_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mujoco-warp==3.5.0",
]

# Installation operation
setup(
    name="mjlab_rl",
    packages=find_namespace_packages(include=["src", "src.*"]),
    version="0.0.1",
    install_requires=INSTALL_REQUIRES,
)
