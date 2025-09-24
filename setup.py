from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whu-supervision-scheduler",
    version="1.0.0",
    author="WHU Supervision Scheduling Team",
    author_email="william.brehm@whu.edu",
    description="A constraint programming-based scheduler for exam supervision at WHU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WilhelmBrehm/whu-supervision-scheduler",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Education",
        "Topic :: Office/Business :: Scheduling",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "ortools>=9.0.0",
        "openpyxl>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "whu-schedule=whu_supervision_scheduler.__main__:main",
        ],
    },
)