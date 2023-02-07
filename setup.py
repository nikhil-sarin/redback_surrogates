from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='redback_surrogates',
      version='0.1',
      description='A surrogates package to work with redback, the bayesian inference package for electromagnetic transients',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/nikhil-sarin/redback_surrogates',
      author='Nikhil Sarin',
      author_email='nikhil.sarin@su.se',
      license='GNU General Public License v3 (GPLv3)',
      packages=['redback_surrogates'],
      package_dir={'redback_surrogates': 'redback_surrogates', },
      package_data={'redback_surrogates': ['surrogate_data/*']},
      python_requires=">=3.7",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",],
      zip_safe=False)
