import codecs
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = "0.0.2"
DESCRIPTION = "A simple package to create elegant nlp pipelines using sklearn."

# Setting up
setup(
    name="pippi-lang",
    version=VERSION,
    author="Szymon Ruciński",
    url="https://github.com/szymonrucinski/pippi-lang",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "nltk",
        "pandas",
        "coloredlogs",
        "beautifulsoup4",
        "lxml",
    ],
    keywords=["python", "stream", "sockets"],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
