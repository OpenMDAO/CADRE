#
# CADRE setup
#

from setuptools import setup

kwargs = {'author': 'Kenneth T. Moore',
 'author_email': 'kenneth.t.moore-1@nasa.gov',
 'classifiers': ['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering'],
 'description': 'Implementation of the CADRE CubeSat design problem for OpenMDAO 1.0 and higher',
 'download_url': 'http://github.com/OpenMDAO/CADRE.git',
 'include_package_data': True,
 'install_requires': ['openmdao'],
 'keywords': ['openmdao', 'CADRE'],
 'license': 'Apache 2.0',
 'maintainer': 'Kenneth T. Moore',
 'maintainer_email': 'kenneth.t.moore-1@nasa.gov',
 'name': 'CADRE',
 'package_data': {'CADRE': ['test/*.pkl']},
 'package_dir': {'': 'src'},
 'packages': ['CADRE', 'CADRE.test'],
 'include_package_data': True,
 'url': 'http://github.com/OpenMDAO/CADRE.git',
 'version': '0.1',
 'zip_safe': False}


setup(**kwargs)

