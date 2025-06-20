# Installation Guide

This guide provides detailed instructions for setting up the Translation Tasks Optimizer project using Conda.

## Prerequisites

Before starting the installation, ensure you have the following installed on your system:

- **Conda**: Either [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- **Git**: For cloning the repository

## Installation Steps

### Clone the Repository

```bash
git clone https://github.com/dmxocean/synthesisOne
cd synthesisOne
```

### Create Conda Environment

Create a new conda environment named `Synthesis` using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

This command will:
- Create a new environment called `Synthesis`
- Install Python and all required dependencies specified in environment.yml
- Set up the environment with all necessary packages for the project

### Activate the Environment

Once the environment is created, activate it:

```bash
conda activate Synthesis
```

You should see `(Synthesis)` prefix in your terminal prompt, indicating the environment is active.

### Verify Installation

To verify the installation was successful:

```bash
python --version
conda list
```

This will display the Python version and all installed packages in the environment.

## Usage

After successful installation:

- Always activate the environment before working on the project:
   ```bash
   conda activate Synthesis
   ```

- To deactivate the environment when finished:
   ```bash
   conda deactivate
   ```

## Updating the Environment

If the environment.yml file is updated with new dependencies:

```bash
conda env update -f environment.yml -p $CONDA_PREFIX
```

## Troubleshooting

### Environment Already Exists

If you get an error that the environment already exists:

```bash
conda env remove -n Synthesis
conda env create -f environment.yml
```

### Package Conflicts

If you encounter package conflicts during installation:

- Try updating conda first:
   ```bash
   conda update conda
   ```

- Then retry the environment creation

### Permission Issues

On Unix-like systems, if you encounter permission issues:
- Ensure you have write permissions in the conda directory
- Consider using `--prefix` to install in a different location

## Additional Notes

- The `Synthesis` environment is specifically configured for this project
- All project scripts should be run within this environment
- The environment.yml file contains all necessary dependencies with tested versions
- For development purposes, you can export your environment changes:
  ```bash
  conda env export > environment.yml
  ```