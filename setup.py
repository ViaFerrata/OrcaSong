#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='orcasong',
    version='1.0',
    description='Makes images for a NN based on the hit information of neutrino events in the neutrino telescope KM3NeT-ORCA',
    url='https://github.com/ViaFerrata/OrcaSong',
    author='Michael Moser',
    author_email='mmoser@km3net.de, michael.m.moser@fau.de',
    license='AGPL',
    packages=find_packages(),
    include_package_data=True,
)