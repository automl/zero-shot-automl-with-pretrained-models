import os
import setuptools

console_scripts = [
    'autofolio=autofolio.autofolio:main'
]

with open("autofolio/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name="autofolio",
    version=version,
    author="Marius Lindauer",
    author_email="lindauer@cs.uni-freiburg.de",
    description=("AutoFolio 2, an automaticalliy configured algorithm selector."),
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
    tests_require=['mock',
                   'nose'],
    test_suite='nose.collector',
    entry_points = {
        'console_scripts': console_scripts
    }
)
