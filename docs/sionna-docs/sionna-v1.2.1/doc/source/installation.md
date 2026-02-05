# Installation

Sionna is composed of three Python modules, namely [Sionna
RT](rt/index.html), [Sionna PHY](phy/index.html), and [Sionna
SYS](sys/index.html).

Sionna PHY and Sionna SYS require [Python
3.10-3.12](https://www.python.org/) and [TensorFlow
2.14-2.19](https://www.tensorflow.org/install). We recommend Ubuntu
24.04. Earlier versions of TensorFlow may still work but are not
recommended. We refer to the [TensorFlow GPU support
tutorial](https://www.tensorflow.org/install/gpu) for GPU support and
the required driver setup.

[Sionna RT](rt/index.html) has the same requirements as [Mitsuba
3](https://github.com/mitsuba-renderer/mitsuba3) and we refer to its
[installation guide](https://mitsuba.readthedocs.io/en/stable/) for
further information. To run Sionna RT on CPU, [LLVM](https://llvm.org)
is required by [Dr.Jit](https://drjit.readthedocs.io/en/stable/). Please
check the [installation instructions for the LLVM
backend](https://drjit.readthedocs.io/en/latest/what.html#backends).

If you want to run the tutorial notebooks on your machine, you also need
[JupyterLab](https://jupyter.org/). You can alternatively test them on
[Google Colab](https://colab.research.google.com/). Although not
necessary, we recommend running Sionna in a [Docker
container](https://www.docker.com) and/or [Python virtual
enviroment](https://docs.python.org/3/library/venv.html).

The [Sionna Research Kit](rk/index.html) runs on the [NVIDIA Jetson AGX
Orin
platform](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/).
We refer to the [quickstart guide](rk/quickstart.html) for a detailed
introduction.

## Using pip

The recommended way to install Sionna is via pip:

``` bash
pip install sionna
```

If you want to install only Sionna RT, run:

``` bash
pip install sionna-rt
```

If you want to install Sionna without the RT package, run:

``` bash
pip install sionna-no-rt
```

## From source

1.  Clone the repository with all submodules:

    > ``` bash
    > git clone --recursive https://github.com/NVlabs/sionna
    > ```
    >
    > If you have already cloned the repository but forgot to set the
    > <span class="title-ref">--recursive</span> flag, you can correct
    > this via:
    >
    > ``` bash
    > git submodule update --init --recursive --remote
    > ```

2.  Install Sionna (including Sionna RT) by running the following
    command from within the repository's root folder:

    > ``` bash
    > pip install ext/sionna-rt/ .
    > pip install .
    > ```

## Testing

First, you need to install the test requirements by executing the
following command from the repository's root directory:

``` bash
pip install '.[test]'
```

The unit tests can then be executed by running `pytest` from within the
`test` folder.

## Documentation

Install the requirements for building the documentation by running the
following command from the repository's root directory:

``` bash
pip install '.[doc]'
```

You might need to install [pandoc](https://pandoc.org) manually.

You can then build the documentation by executing `make html` from
within the `doc` folder.

The documentation can finally be served by any web server, e.g.,

``` bash
python -m http.server --dir build/html
```

## Developing

Development requirements can be installed by executing from the
repository's root directory:

``` bash
pip install '.[dev]'
```

Linting of the code can be achieved by running `pylint src/` from the
repository's root directory.
