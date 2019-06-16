from setuptools import find_packages
from setuptools import setup

setup(name='turbine_opt',
      setup_requires=['setuptools_scm'],
      use_scm_version={'write_to': 'turbine_opt/version.txt'},
      description="app to balance turbine blades",
      packages=find_packages(),
      test_suite = 'tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=[],
      zip_safe=False)
