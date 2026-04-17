"""Tool for clicking elements on a page via the TMWebDriver session."""

from __future__ import annotations

from typing import Any


def click_element(
    session: Any,
    selector: str,
    selector_type: str = "css",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Click a DOM element identified by *selector*.

    Args:
        session: Active :class:`TMWebDriver.Session` instance.
        selector: CSS selector or XPath expression.
        selector_type: ``"css"`` (default) or ``"xpath"``.
        timeout: Seconds to wait for the element to become clickable.

    Returns:
        A dict with keys ``success``, ``selector``, and ``message``.

    Raises:
        RuntimeError: If the session is not active.
        ValueError: If *selector_type* is unsupported.
    """
    if not session.is_active():
        raise RuntimeError("Session is not active; cannot click element.")

    if selector_type not in ("css", "xpath"):
        raise ValueError(f"Unsupported selector_type: {selector_type!r}")

    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    by = By.CSS_SELECTOR if selector_type == "css" else By.XPATH

    driver = session.driver
    element = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((by, selector))
    )
    element.click()

    return {
        "success": True,
        "selector": selector,
        "message": f"Clicked element matching {selector_type} selector: {selector}",
    }
