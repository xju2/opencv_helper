# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

#with open('LICENSE') as f:
#    license = f.read()


setup(name='opencv_helper',
      version='0.1',
      description='helper function of opencv',
      url='https://github.com/xju2/opencv_helper',
      long_description=readme,
      author='Xiangyang Ju',
      author_email='xiangyang.ju@gmail.com',
      #license=license,
      packages=['opencv_helper'],
      zip_safe=False
     )
