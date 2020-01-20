import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn_toolkit",
    version="0.0.0",
    author="Michael Yen",
    author_email="mike.a.yen@gmail.com",
    description="Helpers for training neural nets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mike-a-yen/nn-toolkit",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
