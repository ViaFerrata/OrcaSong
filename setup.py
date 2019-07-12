#!/usr/bin/env python
from setuptools import setup, find_packages
from pkg_resources import get_distribution, DistributionNotFound

with open('requirements.txt') as fobj:
    requirements = [l.strip() for l in fobj.readlines()]

setup(
    name='orcasong',
    description='Makes images for a NN based on the hit information of neutrino events in the neutrino telescope KM3NeT',
    url='https://git.km3net.de/ml/OrcaSong',
    author='Michael Moser, Stefan Reck',
    author_email='mmoser@km3net.de, michael.m.moser@fau.de, stefan.reck@fau.de',
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

    entry_points={'console_scripts': ['make_nn_images=orcasong.make_nn_images:main',
                                      'shuffle=orcasong_contrib.data_tools.shuffle.shuffle_h5:main',
                                      'concatenate=orcasong_contrib.data_tools.concatenate.concatenate_h5:main',
                                      'make_dsplit=orcasong_contrib.data_tools.make_data_split.make_data_split:main',
                                      'plot_binstats=orcasong_2.util.bin_stats_plot:main']}

)

__author__ = 'Michael Moser'