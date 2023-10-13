from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT = '-e .'


def get_requirements(filepath)->List[str]:
    """ THIS FUNCTION GET ALL REQUIREMENTS TO INSTALL"""
    requirements = []

    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements


setup(
name= "Student Performance",
version="0.0.1",
author= "Sai Surya Chandra Prasad",
author_email="saisuryachandraprasad@gmail",
packages=find_packages(),
install_requirements = get_requirements("requirements.txt")
)