from setuptools import find_packages, setup

setup(
    name='pytorch-spear',
    version='0.1.0',
    description='Library for benchmarking adversarial defence methods',
    url='http://github.com/tejus-gupta/spear',
    author='Tejus Gupta',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torchvision',
    ],
    include_package_data=True,
    zip_safe=False,
)