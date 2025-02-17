# Documentation

## Building documentation

1. Install the documentation dependencies
2. Build the html documents under the `docs` directory

Using poetry:

```bash
# Install the optional documentation dependencies
poetry install --extras "doc"
# Make the html documentation
cd docs/source
poetry run sphinx-build . build
# View the documentation
open build/html/index.html
```

Using pip:

```bash
# Install the optional documentation dependencies
pip install .[doc]
# Make the html documentation
make -C docs html
# View the documentation
open docs/build/html/index.html
```

Using make:

```bash
cd docs/
# Cleans up old builds
make clean
# New build
make html
```
