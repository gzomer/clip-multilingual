import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name='clip-multilingual',
    package_dir={'clip_multilingual': ''},
    packages=['clip_multilingual'],
    version='1.0.1',
    description='MultiLingual CLIP - Semantic Image Search in 100 Languages',
    author='Gustavo Zomer',
    nclude_package_data=True,
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), 'requirements.txt'))
        )
    ],
    dependency_links = [
        'git+https://github.com/openai/CLIP.git#egg=clip',
    ],
)