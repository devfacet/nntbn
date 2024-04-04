# Init vars
MAKEFILE := $(lastword $(MAKEFILE_LIST))
BASENAME := $(shell basename "$(PWD)")
SHELL := /bin/bash

.PHONY: help run-arm-examples
all: help
help: Makefile
	@echo
	@echo " Commands:"
	@echo
	@sed -n 's/^##//p' $< | sed -e 's/^/ /' | sort
	@echo

## build-example                       Build an example (e.g., make build-example ARCH=arm EXAMPLE=arch/arm/neon/neuron)
build-example:
	@echo building $(EXAMPLE)
	@ARCH=$(ARCH) EXAMPLE=$(EXAMPLE) scripts/shell/build_example.sh

## build-examples                      Build all arch specific examples (e.g., make build-examples ARCH=arm)
build-examples:
	@$(eval EXAMPLES := $(shell find examples/arch/$(ARCH) -type f -name 'main.c' | sed 's|/main.c||' | sed 's|examples/||' | sed 's|^/||'))
	@for example in $(EXAMPLES); do \
		$(MAKE) build-example ARCH=$(ARCH) EXAMPLE=$$example; \
	done

## run-example                         Run an examples (e.g., make run-example ARCH=arm EXAMPLE=arch/arm/neon/neuron)
run-example:
	@echo running $(EXAMPLE) $(ARGS)
	@ARCH=$(ARCH) EXAMPLE=$(EXAMPLE) ARGS="$(ARGS)" scripts/shell/run_example.sh
	@echo " "

## run-examples                        Run all arch specific examples (e.g., make run-examples ARCH=arm)
run-examples:
	@$(eval EXAMPLES := $(shell find examples/arch/$(ARCH) -type f -name 'main.c' | sed 's|/main.c||' | sed 's|examples/||' | sed 's|^/||'))
	@for example in $(EXAMPLES); do \
		$(MAKE) run-example @ARCH=$(ARCH) EXAMPLE=$$example; \
	done

## clean                               Clean the build directory
clean:
	@if [ -z "$(PWD)" ]; then \
		echo "PWD variable is not set, aborting clean"; \
	elif [ -d "$(PWD)/build" ]; then \
		rm -rf "$(PWD)/build"; \
		echo "build directory removed"; \
	else \
		echo "build directory does not exist, skipping"; \
	fi

## test                                Run tests
test:
	@mkdir -p $(PWD)/build/tests/
	@/usr/bin/gcc -Wall -fdiagnostics-color=always -g $(addprefix -D,$(DEFINES)) \
		-Iinclude/ \
		-Ilib/CMSIS-DSP/Include \
		lib/CMSIS-DSP/Source/BasicMathFunctions/arm_dot_prod_f32.c \
		src/arch/arm/neon/*.c \
		src/arch/arm/cmsis/*.c \
		src/*.c \
		tests/*.c \
		-o $(PWD)/build/tests/tests
	@$(PWD)/build/tests/tests
