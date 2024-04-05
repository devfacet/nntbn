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

## run-test                            Run a test (e.g., make run-test ARCH=generic TEST=arch/generic/neuron)
run-test:
	@echo building $(TEST)
	@ARCH=$(ARCH) TECH=$(TECH) ARTIFACT=tests/$(TEST) scripts/shell/build_artifact.sh
	@echo running $(TEST)
	@ARCH=$(ARCH) TECH=$(TECH) ARTIFACT=tests/$(TEST) ARGS="$(ARGS)" scripts/shell/run_artifact.sh
	@echo " "

## test                                Run tests (e.g., make test ARCH=generic)
test:
	@$(eval TECH_FILTER := $(if $(TECH),$(shell echo $(TECH) | tr ',' '|'),.*)) # Convert TECH to regex filter
	@$(eval TESTS := $(shell find tests/arch/$(ARCH) -type f -name 'main.c' | grep -E "$(TECH_FILTER)" | sed 's|/main.c||' | sed 's|tests/||'))
	@for test in $(TESTS); do \
		$(MAKE) run-test ARCH=$(ARCH) TECH=$(TECH) TEST=$$test; \
	done

## build-example                       Build an example (e.g., make build-example ARCH=generic EXAMPLE=arch/generic/neuron)
build-example:
	@echo building $(EXAMPLE)
	@ARCH=$(ARCH) TECH=$(TECH) ARTIFACT=examples/$(EXAMPLE) scripts/shell/build_artifact.sh

## build-examples                      Build examples (e.g., make build-examples ARCH=generic)
build-examples:
	@$(eval TECH_FILTER := $(if $(TECH),$(shell echo $(TECH) | tr ',' '|'),.*)) # Convert TECH to regex filter
	@$(eval EXAMPLES := $(shell find examples/arch/$(ARCH) -type f -name 'main.c' | grep -E "$(TECH_FILTER)" | sed 's|/main.c||' | sed 's|examples/||'))
	@for example in $(EXAMPLES); do \
		$(MAKE) build-example ARCH=$(ARCH) TECH=$(TECH) EXAMPLE=$$example; \
	done

## run-example                         Run an examples (e.g., make run-example ARCH=generic EXAMPLE=arch/generic/neuron)
run-example:
	@echo running $(EXAMPLE) $(ARGS)
	@ARCH=$(ARCH) ARTIFACT=examples/$(EXAMPLE) ARGS="$(ARGS)" scripts/shell/run_artifact.sh
	@echo " "

# run-examples                         Run examples (e.g., make run-examples ARCH=generic)
run-examples:
	@$(eval TECH_FILTER := $(if $(TECH),$(shell echo $(TECH) | tr ',' '|'),.*)) # Convert TECH to regex filter
	@$(eval EXAMPLES := $(shell find examples/arch/$(ARCH) -type f -name 'main.c' | grep -E "$(TECH_FILTER)" | sed 's|/main.c||' | sed 's|examples/||'))
	@for example in $(EXAMPLES); do \
		$(MAKE) run-example ARCH=$(ARCH) TECH=$(TECH) EXAMPLE=$$example; \
	done
