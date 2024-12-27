from setuptools import setup

setup(
    name="se3_diffusion",
    packages=[
        'se3diff_data',
        'se3diff_analysis',
        'se3diff_model',
        'se3diff_experiments',
        'openfold'
    ],
    package_dir={
        'se3diff_data': './se3diff_data',
        'se3diff_analysis': './se3diff_analysis',
        'se3diff_model': './se3diff_model',
        'se3diff_experiments': './se3diff_experiments',
        'openfold': './openfold',
    },
)
