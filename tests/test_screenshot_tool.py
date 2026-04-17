"""Tests for tools/screenshot_tool.py"""

import base64
import os
import unittest
from unittest.mock import MagicMock, patch

from tools.screenshot_tool import take_screenshot, TOOL_SCHEMA


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # minimal fake PNG bytes


def _make_session(active: bool = True):
    session = MagicMock()
    session.is_active.return_value = active
    session.driver.get_screenshot_as_png.return_value = FAKE_PNG
    return session


class TestTakeScreenshot(unittest.TestCase):
    @patch("tools.screenshot_tool.os.makedirs")
    def test_returns_expected_keys(self, _makedirs):
        session = _make_session()
        with patch("builtins.open", unittest.mock.mock_open()):
            result = take_screenshot(session, save_path="/tmp/test.png")
        self.assertIn("path", result)
        self.assertIn("base64", result)
        self.assertIn("timestamp", result)

    @patch("tools.screenshot_tool.os.makedirs")
    def test_base64_encodes_png(self, _makedirs):
        session = _make_session()
        with patch("builtins.open", unittest.mock.mock_open()):
            result = take_screenshot(session, save_path="/tmp/test.png")
        decoded = base64.b64decode(result["base64"])
        self.assertEqual(decoded, FAKE_PNG)

    def test_raises_when_session_inactive(self):
        session = _make_session(active=False)
        with self.assertRaises(RuntimeError):
            take_screenshot(session)

    @patch("tools.screenshot_tool.os.makedirs")
    def test_auto_generates_save_path(self, _makedirs):
        session = _make_session()
        with patch("builtins.open", unittest.mock.mock_open()):
            result = take_screenshot(session)
        self.assertTrue(result["path"].startswith("screenshots"))
        self.assertTrue(result["path"].endswith(".png"))

    def test_tool_schema_structure(self):
        self.assertEqual(TOOL_SCHEMA["name"], "take_screenshot")
        self.assertIn("description", TOOL_SCHEMA)
        self.assertIn("parameters", TOOL_SCHEMA)


if __name__ == "__main__":
    unittest.main()
