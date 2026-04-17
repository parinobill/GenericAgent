"""Screenshot tool for GenericAgent — captures the current browser state."""

import base64
import os
from datetime import datetime
from typing import Optional


def take_screenshot(session, save_path: Optional[str] = None) -> dict:
    """Take a screenshot of the current browser state.

    Args:
        session: TMWebDriver Session instance.
        save_path: Optional file path to save the PNG. If None, a timestamped
                   file is created in ./screenshots/.

    Returns:
        dict with keys: path (str), base64 (str), timestamp (str)
    """
    if not session.is_active():
        raise RuntimeError("Browser session is not active.")

    os.makedirs("screenshots", exist_ok=True)

    if save_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("screenshots", f"screenshot_{ts}.png")

    png_bytes = session.driver.get_screenshot_as_png()
    with open(save_path, "wb") as f:
        f.write(png_bytes)

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    timestamp = datetime.now().isoformat()

    return {"path": save_path, "base64": b64, "timestamp": timestamp}


# Tool schema consumed by agentmain.load_tool_schema
TOOL_SCHEMA = {
    "name": "take_screenshot",
    "description": (
        "Capture a screenshot of the current browser page. "
        "Returns the file path and a base64-encoded PNG image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "save_path": {
                "type": "string",
                "description": "Optional file path to save the screenshot PNG.",
            }
        },
        "required": [],
    },
}
