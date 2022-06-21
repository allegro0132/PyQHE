from setuptools import setup, find_packages
from os import path

name = "pyqhe"

# for simplicity we actually store the version in the __version__ attribute in the
# source
here = path.abspath(path.dirname(__file__))

MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = True

version = "%d.%d.%d" % (MAJOR, MINOR, MICRO)

author = "Ziang Wang"

author_email = "ziangwang9@zju.edu.cn"

description = "PyQHE"

url = "https://github.com/allegro0132/pyqhe"

license = "Apache-2.0"

classifiers = [
    "Intended Audience :: Developers", "Intended Audience :: Science/Research",
    "Operating System :: Microsoft :: Windows", "Operating System :: MacOS",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8", "Topic :: Scientific/Engineering",
    "Environment :: Console",
    "License :: OSI Approved :: Apache Software License"
],


setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    maintainer=author,
    maintainer_email=author_email,
    description=description,
    url=url,
    license=license,
    classifiers=classifiers,
    packages=find_packages(),
    include_package_data=True
)
