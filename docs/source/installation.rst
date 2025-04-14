Installation
============

You can install the package directly from PyPI:

.. code-block:: bash

    pip install langchain-aerospike

Development Installation
------------------------

This project uses Poetry for dependency management and packaging. To set up your development environment:

1. Install Poetry if you haven't already:

.. code-block:: bash

    curl -sSL https://install.python-poetry.org | python3 -

2. Clone the repository and install dependencies:

.. code-block:: bash

    git clone https://github.com/aerospike/langchain-aerospike.git
    cd langchain-aerospike
    poetry install

3. Activate the virtual environment:

.. code-block:: bash

    eval $(poetry env activate)

Running Examples
----------------

The examples in the ``examples/`` directory require additional dependencies. You can install them with:

.. code-block:: bash

    poetry install --with examples

This will install dependencies like ``langchain-huggingface`` which are used in the example scripts. 