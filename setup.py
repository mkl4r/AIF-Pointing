from setuptools import setup, find_packages

setup(
   name='aif-pointing',
   version='1.0',
   author='Markus Klar',
   author_email='markus.klar@glasgow.ac.uk',
   packages=['aif-pointing'],
   package_data={'': ['**']},
   url='https://github.com/mkl4r/AIF-Pointing',
   license='LICENSE',
   python_requires='>=3.11',
   install_requires=[
       "numpy", "matplotlib", "pyyaml", "optax", "tqdm", "difai @ git+https://github.com/mkl4r/difai-base.git@main"
   ],
   extras_require={
    "gpu": ["jax[cuda13]"],
    "tpu": ["jax[tpu]"],
    "cpu": ["jax"],
    },
)