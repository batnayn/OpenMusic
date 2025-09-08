"""
Setup script for OpenMusic AI Studio
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="openmusic",
    version="0.1.0",
    author="OpenMusic Team",
    author_email="team@openmusic.ai",
    description="AI Music Studio for creating music and vocals using open-source models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/batnayn/OpenMusic",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "advanced": [
            "audiocraft @ git+https://github.com/facebookresearch/audiocraft.git",
            "bark @ git+https://github.com/suno-ai/bark.git",
        ],
        "dev": [
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
        ]
    },
    entry_points={
        "console_scripts": [
            "openmusic=app.main:main",
        ],
    },
    keywords="ai music generation tts vocals llm audio",
    project_urls={
        "Bug Reports": "https://github.com/batnayn/OpenMusic/issues",
        "Source": "https://github.com/batnayn/OpenMusic",
        "Documentation": "https://github.com/batnayn/OpenMusic#readme",
    },
)