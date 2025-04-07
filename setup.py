#we'll set entire app as package 
from setuptools import setup,find_packages
from typing import List

Hyphen_E_DOT='-e.'
def get_requirements(file_path:str)->List[str]:
    '''This function will return the list of req '''  
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")  for req in requirements]

        if Hyphen_E_DOT in requirements:
            requirements.remove(Hyphen_E_DOT) 
    return requirements
            




setup(
    name='UberRidesCancellationPrediction',
    version='0.0.1',
    author="Vishi",
    author_email='vishiiikant@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    ) 