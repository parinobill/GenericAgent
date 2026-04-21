"""Tool for typing text into a focused or targeted element in the browser session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException
from selenium.webdriver.common.by import By

if TYPE_CHECKING:
    from TMWebDriver import Session


# Mapping of selector type strings to Selenium By constants
SELECTOR_TYPE_MAP = {
    "css": By.CSS_SELECTOR,
    "xpath": By.XPATH,
    "id": By.ID,
    "name": By.NAME,
}


def type_text(
    session: "Session",
    selector_type: str,
    selector: str,
    text: str,
    clear_first: bool = True,
) -> dict:
    """Type text into an element located by the given selector.

    Args:
        session: Active TMWebDriver session.
        selector_type: One of 'css', 'xpath', 'id', or 'name'.
        selector: The selector string used to locate the target element.
        text: The text to type into the element.
        clear_first: If True, clears any existing text in the element before typing.
                     Defaults to True.

    Returns:
        A dict with:
            - 'success' (bool): Whether the action succeeded.
            - 'message' (str): Human-readable description of the outcome.

    Raises:
        RuntimeError: If the session is not active.
        ValueError: If selector_type is not a recognised value.
        NoSuchElementException: If no element matches the selector.
        ElementNotInteractableException: If the element cannot receive input.
    """
    if not session.is_active():
        raise RuntimeError("Browser session is not active. Cannot type text.")

    selector_type_lower = selector_type.lower()
    if selector_type_lower not in SELECTOR_TYPE_MAP:
        raise ValueError(
            f"Invalid selector_type '{selector_type}'. "
            f"Must be one of: {', '.join(SELECTOR_TYPE_MAP.keys())}"
        )

    by = SELECTOR_TYPE_MAP[selector_type_lower]

    try:
        element = session.driver.find_element(by, selector)
    except NoSuchElementException:
        raise NoSuchElementException(
            f"No element found with {selector_type}='{selector}'."
        )

    try:
        if clear_first:
            element.clear()
        element.send_keys(text)
    except ElementNotInteractableException:
        raise ElementNotInteractableException(
            f"Element with {selector_type}='{selector}' is not interactable."
        )

    return {
        "success": True,
        "message": f"Typed text into element with {selector_type}='{selector}'.",
    }
