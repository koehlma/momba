# Momba

A Python library for *quantitative models*.


## Development
This project uses [Pipenv](https://pipenv.kennethreitz.org/) for dependency management. Run `pipenv install --dev` to create a virtual environment in `.venv` containing all the dependencies needed for development. The project comes with a configuration for [Visual Studio Code](https://code.visualstudio.com/) which requires the virtual environment and enables linting and type checking. Before *pushing* ensure that `pipenv run tox` runs without any problems.
