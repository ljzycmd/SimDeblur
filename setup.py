from setuptools import setup, find_packages

VERSION = "0.1.0"

setup(
    name="simdeblur",
    version=VERSION,
    description="A simple deblurring framework for image/video deblurring tasks.",
    author="Mingdeng Cao",
    author_email="mingdengcao@gmail.com",
    url="https://github.com/ljzycmd/SimDeblur",
    license="MIT",
    packages=find_packages(),
    zip_safe=True
)
