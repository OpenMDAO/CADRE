#
# CADRE setup
#

from setuptools import setup

kwargs = {
    'name': 'CADRE',
    'description': 'Implementation of the CADRE CubeSat design problem for OpenMDAO 2.x',
    'license': 'Apache 2.0',
    'author': 'Kenneth T. Moore',
    'author_email': 'kenneth.t.moore-1@nasa.gov',
    'maintainer': 'Kenneth T. Moore',
    'maintainer_email': 'kenneth.t.moore-1@nasa.gov',
    'classifiers': [
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    'keywords': ['openmdao', 'CADRE'],
    'url': 'http://github.com/OpenMDAO/CADRE.git',
    'download_url': 'http://github.com/OpenMDAO/CADRE.git',
    'install_requires': ['openmdao>=2.3'],
    'packages': ['CADRE', 'CADRE.test'],
    'package_data': {'CADRE': ['data/*.pkl', 'test/*.pkl']},
    'include_package_data': True,
    'version': '0.2',
    'zip_safe': False
}

setup(**kwargs)
