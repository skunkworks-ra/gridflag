"""Tests for gridflag.utils."""

from __future__ import annotations

import logging

from gridflag.utils import setup_logging


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("WARNING")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "gridflag"

    def test_level_applied(self):
        logger = setup_logging("DEBUG")
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        """Calling setup_logging twice should not add a second handler."""
        logger = setup_logging("INFO")
        n = len(logger.handlers)
        setup_logging("DEBUG")
        assert len(logger.handlers) == n

    def test_case_insensitive(self):
        logger = setup_logging("warning")
        assert logger.level == logging.WARNING
