# ABOUTME: pytest configuration for the sw toolchain tests.
# ABOUTME: Registers the `slow` marker used by the full-resolution equivalence test.

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the full-resolution model or a network fetch")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
