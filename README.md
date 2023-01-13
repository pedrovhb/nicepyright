## nicepyright

nicepyright is a tool that provides a nicer CLI interface to the pyright type checker. It continuously monitors and displays type warnings in a more user-friendly format.

Please note that nicepyright is currently in an early stage of development, so it may be feature incomplete and contain bugs. However, it is already useful and can help you to find type warnings in your code faster and more easily.

### Installation

`nicepyright` is available on PyPI and can be installed with `pip`, `poetry`, or your favorite Python package manager.

```bash
poetry add --dev nicepyright
```

### Usage

To use nicepyright, navigate to the root directory of your project and run the following command:

```bash
nicepyright
```

Make sure that the environment being used is the one that contains all the libraries your project uses.
That is, if you are using a virtual environment, make sure that it is activated.
If you are using `poetry`, you can use the `poetry run` command to ensure that the correct version of `nicepyright` is used.

```bash
poetry run nicepyright
```

This will start the pyright type checker and display the type warnings in a more user-friendly format.

