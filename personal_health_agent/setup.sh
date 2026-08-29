#!/bin/bash

conda_setup() {
    echo "Setting up pha_framework environment using conda..."
    
    # Remove existing pha_framework environment if it exists
    conda env remove -n pha_framework -y || echo "No existing pha_framework environment found"
    
    # Create and activate initial pha_framework environment with Python 3.12
    conda create -n pha_framework python=3.12 pip -y || exit 1
    
    # Source conda to ensure 'conda activate' works within the script
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate pha_framework || exit 1
    
    # Verify pip and python info for debugging
    echo "Using pip from: $(which pip)"
    echo "Python version: $(python --version)"
    
    # Install standard packages from requirements.txt
    echo "Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt || exit 1
    
    # Install OneTwo from the official Google DeepMind repo
    # Using --no-deps to prevent it from overwriting our specific versions of shared libs (like numpy)
    echo "Installing OneTwo..."
    python -m pip install --no-deps git+https://github.com/google-deepmind/onetwo || exit 1
    
    # Register the kernel with Jupyter
    echo "Registering Jupyter kernel..."
    python -m ipykernel install --user --name pha_framework --display-name "pha_framework (Python 3.12)" || exit 1
    
    echo "----------------------------------------------------------------"
    echo "Setup complete! Environment 'pha_framework' is ready."
    echo "To start using it, run: conda activate pha_framework"
    echo "----------------------------------------------------------------"
}

conda_setup
