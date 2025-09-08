"""OpenMusic - Comprehensive Audio Processing Library"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="openmusic",
    version="1.0.0",
    author="OpenMusic Team",
    author_email="team@openmusic.org",
    description="Comprehensive audio processing library with 1000+ features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/batnayn/OpenMusic",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "speechrecognition>=3.8.0",
        "pyttsx3>=2.90",
        "noisereduce>=2.0.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
)