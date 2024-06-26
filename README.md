# nntbn - A neural network library

nntbn is a neural network library to be named.
The goal of this library is to provide a framework for training and running neural networks
across various platforms, from server-grade hardware to desktops, mobile, and embedded devices.

## ⚠️️ This project is in the early stages of development and is not yet ready for use.

## Requirements

- [Clang](https://clang.llvm.org/)
- [Make](https://www.gnu.org/software/make/)

### Recommended tools

- [Visual Studio Code](https://code.visualstudio.com/)
- [C/C++ for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)

## Installation

```shell
git clone https://github.com/devfacet/nntbn.git
cd nntbn/
git submodule update --init
```

## Usage

## Test

```shell
# Test the default architecture (generic)
make test

# Test a specific architecture
make test ARCH=generic
make test ARCH=arm

# Test a specific architecture with filters
make test ARCH=generic FILTERS=dot_prod
make test ARCH=arm FILTERS=neon,cmsis-dsp

# Test all architectures
make test-all
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Licensed under The MIT License (MIT)  
For the full copyright and license information, please view the LICENSE.txt file.
