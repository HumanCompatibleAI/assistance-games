from setuptools import find_packages, setup

__version__ = '0.0.1'

setup(
    name='assistance_games',
    version=__version__,
    description='Supporting code for Benefits of Assistance Games over Reward Learning paper',
    author='Center for Human-Compatible AI',
    author_email='pedrofreirex@gmail.com',
    url='https://github.com/HumanCompatibleAI/assistance-games',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        '': [
            'assets/*',
        ],
    },
    install_requires=[
        'numpy>=1.13',
        'scipy>=0.19',
        'sparse>=0.9.1',
        'pyglet',
        'gym',
        'lark-parser>=0.8',
        'stable-baselines>=2.9',
        # 'tensorflow>=1.15.0,<2.0',
    ],
    tests_require=['pytest'],
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
)
