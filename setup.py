from setuptools import setup, find_packages

setup(
    name='rethink',
    version='0.0.1',
    packages=find_packages(include=['rethink', 'rethink.*']),
    install_requires=[
        'torch',
        'transformers',
        'tqdm',
    ]
)