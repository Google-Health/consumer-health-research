#!/bin/bash

conda_setup() {
    echo "Setting up phia environment using conda..."
    
    # Remove existing phia environment if it exists
    conda env remove -n phia -y || echo "No existing phia environment found"
    
    # Create and activate initial phia environment
    conda create -n phia python=3.11 pip -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate phia || exit 1
    
    # Verify pip and python info for debugging
    echo "Using pip from: $(which pip)"
    echo "Python version: $(python --version)"
    
    # Install other packages from requirements.txt first
    python -m pip install -r requirements.txt || exit 1
    
    # Need --no-deps flag to make installation across different machine architectures easier
    python -m pip install --no-deps git+https://github.com/google-deepmind/onetwo || exit 1
    
    # Register the kernel with Jupyter (now that ipykernel is installed)
    python -m ipykernel install --user --name phia --display-name "phia (Python 3.11.x)" || exit 1
    
    echo "Setup complete! Environment 'phia' is ready to use!"
}

conda_setup
