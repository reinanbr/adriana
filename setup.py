from setuptools import setup
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    readme = fh.read()

setup(name='adrianna',
    version='0.0.7.1',
    url='https://github.com/reinanbr/adriana',
    license='MIT License',
    author='Reinan Br',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='slimchatuba@gmail.com',
    keywords='ia machine learning math',
    description=u'lib for development with machine learning',
    packages=find_packages(),
    install_requires=['numpy','scipy'],)
