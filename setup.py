import setuptools

setuptools.setup(
    name="riter",
    version="0.0.1",
    author="Jin Yong Yoo",
    author_email="jy2ma@virginia.edu",
    description="A library image-text retreival",
    license="MIT",
    long_description="A library image-text retreival",
    url="https://github.com/jinyongyoo/RITER",
    packages=setuptools.find_namespace_packages(
        exclude=[
            "build*",
            "dist*",
        ]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").readlines(),
)
