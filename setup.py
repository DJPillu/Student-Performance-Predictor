from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str) -> List[str]:
    '''
    read in packages from file and return as list
    '''
    
    requirements = list()
    with open(file_path) as file:
        # strip as readlines appends a newline character at the end
        requirements = [package.strip() for package in file.readlines()]
        
        # remove directive to build packages using setup.py in requirements list
        requirements.pop(-1)
        
    return requirements

setup(
    name = 'Student Performance Predictor',
    version = "0.0.1",
    author = 'Rujul Arora',
    author_email = "rujularoracs@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
    
)