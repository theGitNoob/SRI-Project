#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found. Installing Poetry..."
    
    # Install Poetry using the official install script
    curl -sSL https://install.python-poetry.org | python3 -

    # Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v poetry &> /dev/null
    then
        echo "Poetry installation failed. Please install it manually."
        exit 1
    fi
else
    echo "Poetry is already installed."
fi

# Install dependencies
echo "Installing dependencies..."
poetry install


# Run the project
echo "Running the project..."
poetry run python sri_project/main.py
