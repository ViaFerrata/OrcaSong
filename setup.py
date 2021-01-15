#!/usr/bin/env python
from setuptools import setup, find_packages
# from pkg_resources import get_distribution, DistributionNotFound

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(
    name='orcasong',
    description='Makes images for a NN based on the hit information of neutrino '
                'events in the neutrino telescope KM3NeT',
    url='https://git.km3net.de/ml/OrcaSong',
    author='Stefan Reck, Michael Moser',
    author_email='stefan.reck@fau.de, mmoser@km3net.de, michael.m.moser@fau.de',
    license='AGPL',
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python', ],

    setup_requires=['setuptools_scm'],
    use_scm_version={'write_to': 'orcasong/version.txt',
                     'tag_regex': r'^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$', },

    entry_points={'console_scripts': [
        'concatenate=orcasong.tools.concatenate:main',
        'h5shuffle=orcasong.tools.postproc:h5shuffle',
        'h5shuffle2=orcasong.tools.shuffle2:h5shuffle2',
        'plot_binstats=orcasong.plotting.plot_binstats:main',
        'make_nn_images=legacy.make_nn_images:main',
        'make_dsplit=orcasong_contrib.data_tools.make_data_split.make_data_split:main']}
)

__author__ = 'Stefan Reck, Michael Moser, Daniel Guderian'
