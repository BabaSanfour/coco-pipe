
from setuptools import setup, find_packages

setup(
    name='cocopipe',
    version='0.0.1',
    author='Hamza Abdelhedi',
    author_email='hamza.abdelhedii@gmail.com',
    description='A pipeline for EEG bio-data analysis with ML and Deep Learning components',
    packages=find_packages(),
    install_requires=[
        'pyyaml',
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'run-coco-pipe=scripts.run_pipeline:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.9',
)
