import os
import setuptools


requirements = ['numpy', 'scipy', 'pyyaml', 'liac-arff', 'pandas']

setuptools.setup(
    name="aslib_scenario",
    version="1.0.0",
    author="Marius Lindauer",
    author_email="lindauer@cs.uni-freiburg.de",
    description=("Python Package to read scenario from the algorithm selection library"),
    license="2-clause BSD",
    keywords="algortithm selection",
    url="",
    packages=setuptools.find_packages(exclude=['test', 'source']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: 2-clause BSD",
    ],
    platforms=['Linux'],
    install_requires=requirements,
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector'
)
