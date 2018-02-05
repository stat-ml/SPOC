from setuptools import setup

setup(name='spoc',
      version='1.0',
      description='Successive projection overlapping clustering algorithm',
      url='https://github.com/premolab/SPOC',
      author='Panov, Slavnov, Mokrov, Ushakov',
      author_email='panov.maxim@gmail.com ',
      license='MIT',
      packages=['spoc'],
      install_requires=[
          'numpy>=1.13.1',
          'scipy>=0.19.1',
          'cvxpy>=0.4.11',
          'cvxopt>=1.1.9',
      ],
      zip_safe=False)
