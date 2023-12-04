from setuptools import setup, find_packages

setup(
    name='optools',
    version='0.1.0',
    packages=find_packages(),
    description="FX option pricing toolbox, based on Foreign Exchange Option Pricing: A Practitioner's Guide by Clark.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Igor Pozdeev',
    author_email='igor.posdeev@gmail.com',
    url='https://github.com/ipozdeev/optools',
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib",
        "statsmodels"
    ],
)