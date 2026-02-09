from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="business-card-ocr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    python_requires='>=3.8,<3.11',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'business-card-ocr=app:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Business Card OCR and Information Extractor",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/business-card-ocr",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)