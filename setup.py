from setuptools import setup, find_packages

setup(
    name='optosim',
    version='0.1',
    author=['Auke Pieter Colijn', 'Marjolein van Nuland Troost', 'Carlo Fuselli'],
    author_email='acolijn@nikhef.nl',
    description='A package for optical simulations of a TPC and related machine learning tools for position reconstruction.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/acolijn/OptoSim',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').readlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)