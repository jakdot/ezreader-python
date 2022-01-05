from setuptools import setup

VERSION = '0.1'

setup(name='ezreader',
      version=VERSION,
      description='E-Z Reader',
      url='https://github.com/jakdot/ezreader-python',
      author='jakdot',
      author_email='j.dotlacil@gmail.com',
      packages=['ezreader'],
      license='GPL',
      install_requires=['numpy', 'simpy'],
      classifiers=['Programming Language :: Python :: 3', 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', 'Operating System :: OS Independent', 'Development Status :: 3 - Alpha', 'Topic :: Scientific/Engineering'],
      zip_safe=False)
