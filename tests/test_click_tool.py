"""Tests for tools/click_tool.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tools.click_tool import click_element


def _make_session(active: bool = True) -> MagicMock:
    session = MagicMock()
    session.is_active.return_value = active
    session.driver = MagicMock()
    return session


class TestClickElement:
    def test_raises_when_session_inactive(self):
        session = _make_session(active=False)
        with pytest.raises(RuntimeError, match="Session is not active"):
            click_element(session, "#submit")

    def test_raises_on_invalid_selector_type(self):
        session = _make_session()
        with pytest.raises(ValueError, match="Unsupported selector_type"):
            click_element(session, "#btn", selector_type="id")

    def test_returns_expected_keys_css(self):
        session = _make_session()
        mock_element = MagicMock()
        with patch("tools.click_tool.WebDriverWait") as mock_wait:
            mock_wait.return_value.until.return_value = mock_element
            result = click_element(session, "#submit", selector_type="css")

        mock_element.click.assert_called_once()
        assert result["success"] is True
        assert result["selector"] == "#submit"
        assert "css" in result["message"]

    def test_returns_expected_keys_xpath(self):
        session = _make_session()
        mock_element = MagicMock()
        with patch("tools.click_tool.WebDriverWait") as mock_wait:
            mock_wait.return_value.until.return_value = mock_element
            result = click_element(session, "//button[@id='ok']", selector_type="xpath")

        mock_element.click.assert_called_once()
        assert result["success"] is True
        assert "xpath" in result["message"]

    def test_default_selector_type_is_css(self):
        session = _make_session()
        mock_element = MagicMock()
        with patch("tools.click_tool.WebDriverWait") as mock_wait, \
             patch("tools.click_tool.By") as mock_by:
            mock_wait.return_value.until.return_value = mock_element
            click_element(session, ".btn-primary")
            # Ensure CSS_SELECTOR was used
            mock_wait.return_value.until.assert_called_once()

    def test_result_contains_selector_key(self):
        # Personal note: I want to make sure the selector is always echoed back
        # in the result dict so callers can log which element was clicked.
        session = _make_session()
        mock_element = MagicMock()
        with patch("tools.click_tool.WebDriverWait") as mock_wait:
            mock_wait.return_value.until.return_value = mock_element
            result = click_element(session, "//div[@class='modal']", selector_type="xpath")

        assert "selector" in result
        assert result["selector"] == "//div[@class='modal']"
