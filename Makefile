# Init vars
MAKEFILE := $(lastword $(MAKEFILE_LIST))
BASENAME := $(shell basename "$(PWD)")
SHELL := /bin/bash
ifeq ($(ARCH),)
ARCH := generic
endif
export ARCH

.PHONY: help run-arm-examples
all: help
help: Makefile
	@echo
	@echo " Commands:"
	@echo
	@sed -n 's/^##//p' $< | sed -e 's/^/ /' | sort
	@echo

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

## run-test                            Run a test (e.g., make run-test ARCH=arm TEST=arch/generic/neuron)
run-test:
	@echo running $(TEST)
	@ARCH=$(ARCH) TEST=$(TEST) scripts/shell/run_test.sh
	@echo " "

## test                                Run tests (e.g., make test ARCH=generic)
test:
	@$(eval TESTS := $(shell find tests/arch/$(ARCH) -type f -name 'main.c' | sed 's|/main.c||' | sed 's|tests/||' | sed 's|^/||'))
	@for test in $(TESTS); do \
		$(MAKE) run-test @ARCH=$(ARCH) TEST=$$test; \
	done

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
