"""
ioos_pkg_skeleton

My awesome ioos_pkg_skeleton
"""

import numpy as np
import requests


def meaning_of_life(n: int) -> np.ndarray:
    """Return the meaning of life n times."""
    matrix = (n, n)
    return np.ones(matrix) * 42


def meaning_of_life_url() -> str:
    """
    Fetch the meaning of life from http://en.wikipedia.org.
    """
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/Monty_Python's_The_Meaning_of_Life"
    r = requests.get(url)
    r.raise_for_status()
    j = r.json()
    return j["extract"]
