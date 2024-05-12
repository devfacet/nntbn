# Init vars
MAKEFILE := $(lastword $(MAKEFILE_LIST))
BASENAME := $(shell basename "$(PWD)")
SHELL := /bin/bash
ifeq ($(ARCH),)
ARCH := generic
endif
export ARCH

.PHONY: help
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

## test                                Run tests (e.g., make test ARCH=generic OR make test ARCH=arm FILTERS=neon,cmsis-dsp)
test:
	@echo testing ARCH=$(ARCH) FILTERS=$(FILTERS) ARGS=$(ARGS)
	@$(MAKE) build-artifacts PREFIX=tests/arch/$(ARCH) FILTERS=$(FILTERS)
	@echo " "
	@$(MAKE) run-artifacts PREFIX=tests/arch/$(ARCH) FILTERS=$(FILTERS) ARGS="$(ARGS)"

## test-all                            Run tests (e.g., make test-all ARCH=generic)
test-all:
	@$(MAKE) build-artifacts PREFIX=tests/arch/generic
	@$(MAKE) build-artifacts PREFIX=tests/arch/arm FILTERS=neon,cmsis-dsp
	@echo " "
	@$(MAKE) run-artifacts PREFIX=tests/arch/generic
	@$(MAKE) run-artifacts PREFIX=tests/arch/arm FILTERS=neon,cmsis-dsp

## build-artifact                      Build an artifact (e.g., make build-artifact ARTIFACT=tests/arch/generic/layer)
build-artifact:
	@echo building ARTIFACT=$(ARTIFACT)
	@ARTIFACT=$(ARTIFACT) scripts/shell/build_artifact.sh

## build-artifacts                     Build artifacts (e.g., make build-artifacts PREFIX=tests/arch/generic)
build-artifacts:
	@$(eval FILTERS_GREP := $(if $(FILTERS),$(shell echo $(FILTERS) | tr ',' '|'),.*))
	@$(eval ARTIFACTS := $(shell find $(PREFIX) -type f -name 'main.c' | grep -E "$(FILTERS_GREP)" | sed 's|/main.c||' | sort))
	@for artifact in $(ARTIFACTS); do \
		$(MAKE) build-artifact ARTIFACT=$$artifact || exit 1; \
	done

## run-artifact                        Run an artifact (e.g., make run-artifact ARTIFACT=tests/arch/generic/layer)
run-artifact:
	@echo running ARTIFACT=$(ARTIFACT) ARGS=$(ARGS)
	@ARTIFACT=$(ARTIFACT) ARGS="$(ARGS)" scripts/shell/run_artifact.sh
	@echo " "

## run-artifacts                       Run tests (e.g., make run-artifacts PREFIX=tests/arch/generic)
run-artifacts:
	@$(eval FILTERS_GREP := $(if $(FILTERS),$(shell echo $(FILTERS) | tr ',' '|'),.*))
	@$(eval ARTIFACTS := $(shell find $(PREFIX) -type f -name 'main.c' | grep -E "$(FILTERS_GREP)" | sed 's|/main.c||' | sort))
	@for artifact in $(ARTIFACTS); do \
		$(MAKE) run-artifact ARTIFACT=$$artifact ARGS="$(ARGS)" || exit 1; \
	done
