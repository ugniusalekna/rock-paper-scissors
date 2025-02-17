from setuptools import setup, find_packages

setup(
    name='mini-dl-workflow',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "ipykernel",
        "ipython",
        "matplotlib",
        "numpy",
        "onnx",
        "onnxruntime",
        "opencv-python",
        "pillow",
        "PyYAML",
        "tensorboard",
        "torch",
        "torchvision",
        "tqdm",
    ],
    extras_require={
        "jupyter": [
            "jupyter",
            "notebook",
            "jupyterlab",
        ]
    },
)