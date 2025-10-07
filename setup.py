from setuptools import setup, find_packages

setup(
    name="musedfm",
    version="1.0.0",
    description="MUSED-FM: Multi-Scale Universal Synthetic Data for Forecasting Models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "pyarrow>=10.0.0",
        "pydantic>=2.0.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
)
