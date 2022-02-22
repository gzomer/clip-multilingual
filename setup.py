import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name='clip-multilingual',
    package_dir={'clip_multilingual': ''},
    packages=['clip_multilingual'],
    version='1.0',
    description='MultiLingual CLIP',
    author='Gustavo Zomer',
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
        )
    ],
)