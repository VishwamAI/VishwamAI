# Makefile for VishwamAI testing infrastructure

# Default variables
DURATION ?= 1
REPORT_DIR ?= test_reports
VIZ_DIR ?= visualizations
PLATFORM ?= all
TEST_TYPE ?= all

# Colors
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m

# Commands
.PHONY: test setup clean test-tpu test-gpu test-cpu report visualize resources-up resources-down estimate help

help:
	@echo "VishwamAI Testing Infrastructure"
	@echo "==============================="
	@echo
	@echo "Available commands:"
	@echo "  make test              Run full test suite across all platforms"
	@echo "  make test-tpu         Run TPU-specific tests"
	@echo "  make test-gpu         Run GPU-specific tests"
	@echo "  make test-cpu         Run CPU-specific tests"
	@echo "  make setup            Setup test environment"
	@echo "  make resources-up     Create cloud resources"
	@echo "  make resources-down   Delete cloud resources"
	@echo "  make report           Generate test report"
	@echo "  make visualize        Create visualizations"
	@echo "  make estimate         Show cost estimate"
	@echo "  make clean            Clean generated files"
	@echo
	@echo "Options:"
	@echo "  DURATION=hours        Test duration in hours (default: 1)"
	@echo "  REPORT_DIR=dir        Report directory (default: test_reports)"
	@echo "  VIZ_DIR=dir          Visualization directory (default: visualizations)"
	@echo "  PLATFORM=platform     Target platform [tpu|gpu|cpu|all] (default: all)"
	@echo "  TEST_TYPE=type        Test type [unit|integration|all] (default: all)"

# Full test workflow
test:
	@echo "${GREEN}Running full test workflow${NC}"
	./run_full_test.sh --duration $(DURATION) --report-dir $(REPORT_DIR) --viz-dir $(VIZ_DIR)

# Platform-specific tests
test-tpu:
	@echo "${GREEN}Running TPU tests${NC}"
	./tests/run_tests.py --platform tpu --test-type $(TEST_TYPE)

test-gpu:
	@echo "${GREEN}Running GPU tests${NC}"
	./tests/run_tests.py --platform gpu --test-type $(TEST_TYPE)

test-cpu:
	@echo "${GREEN}Running CPU tests${NC}"
	./tests/run_tests.py --platform cpu --test-type $(TEST_TYPE)

# Environment setup
setup:
	@echo "${YELLOW}Setting up test environment${NC}"
	./setup_test_env.sh $(PLATFORM)

# Resource management
resources-up:
	@echo "${YELLOW}Creating cloud resources${NC}"
	./manage_test_resources.sh create $(PLATFORM)

resources-down:
	@echo "${YELLOW}Deleting cloud resources${NC}"
	./manage_test_resources.sh delete $(PLATFORM)

# Report generation
report:
	@echo "${YELLOW}Generating test report${NC}"
	./generate_test_report.py \
		--benchmark-file "$(REPORT_DIR)/latest_results.json" \
		--duration $(DURATION) \
		--output "$(REPORT_DIR)/latest_report.md"

# Visualization
visualize:
	@echo "${YELLOW}Creating visualizations${NC}"
	./visualize_test_results.py \
		--results "$(REPORT_DIR)/latest_results.json" \
		--output-dir "$(VIZ_DIR)"

# Cost estimation
estimate:
	@echo "${YELLOW}Generating cost estimate${NC}"
	./manage_test_resources.sh estimate $(PLATFORM) $(DURATION)

# Cleanup
clean:
	@echo "${YELLOW}Cleaning up${NC}"
	rm -rf $(REPORT_DIR)/*
	rm -rf $(VIZ_DIR)/*
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".benchmarks" -exec rm -r {} +

# Initialize directories
$(REPORT_DIR):
	mkdir -p $(REPORT_DIR)

$(VIZ_DIR):
	mkdir -p $(VIZ_DIR)

# Create directories if they don't exist
test report visualize: | $(REPORT_DIR) $(VIZ_DIR)
