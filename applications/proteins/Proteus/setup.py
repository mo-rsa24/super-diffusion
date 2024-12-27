from setuptools import setup

setup(
    name="proteus",
    packages=[
        'proteus_data',
        'proteus_analysis',
        'proteus_model',
        'proteus_experiments',
        'proteus_openfold',
    ],
    package_dir={
        'proteus_data': './proteus_data',
        'proteus_analysis': './proteus_analysis',
        'proteus_model': './proteus_model',
        'proteus_experiments': './proteus_experiments',
        'proteus_openfold': './proteus_openfold',
    },
)
