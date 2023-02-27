import setuptools

# upload to pip
# pip install .
# python3 setup.py sdist bdist_wheel
# twine upload dist/pydoppler-0.1.8.tar.gz

setuptools.setup(
     name='PyROA',
     version='3.1.0',
     packages=['PyROA'] ,
     author="Fergus Donnan,",
     author_email="fergus.donnan@physics.ox.ac.uk",
     description="PyROA is a tool for modelling quasar lightcurves",
   long_description_content_type="text/markdown",
     url="https://github.com/FergusDonnan/PyROA",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
         ],
 )