# Introduction to Gaussian Processes 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/endrelaszlo/IntroGaussianProcess/main?filepath=GaussianProcess.ipynb)

One may start with the tutorial using either:

- Jupyter Notebook served on Binder
- Jupyter Notebook served by own server 
    - Using Docker
    - Using Conda 

## Binder

The Jupyter Notebook is available on Binder for ease of use: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/endrelaszlo/IntroGaussianProcess/main?filepath=GaussianProcess.ipynb)

Note: May take a while to start the notebook as the Docker image might need to be rebuilt.

If you're interested in the Binder Project please visit the project website [MyBinder](https://mybinder.org/) or a brief description about the project on [Jupyter](https://jupyter.org/binder).

## Serving the Notebook using Docker environment

```bash
make build
make run
```

## Serving the Notebook using Conda environment

Clone Github repo and create Conda environment

```bash
git clone https://github.com/endrelaszlo/IntroGaussianProcess.git
cd IntroGaussianProcess 
conda env create -f environment.yml
conda activate gp
juptyer notebook
```

