from typing import Tuple

from numpy import (
    arange,
    array,
    ndarray,
    where,
)

__all__ = [
    "S_BOX",
    "calculate_sbox",
]

S_BOX = array([
    0xf, 0x6, 0x5, 0x8,
    0xe, 0xb, 0xa, 0x4,
    0xc, 0x0, 0x3, 0x7,
    0x2, 0x9, 0x1, 0xd,
], dtype="uint16")

_S_BOX_FRAGMENT = 4


# S_BOX = array([
#     0xe, 0x4, 0xd, 0x1,
#     0x2, 0xf, 0xb, 0x8,
#     0x3, 0xa, 0x6, 0xc,
#     0x5, 0x9, 0x0, 0x7,
# ], dtype="uint16")


def calculate_sbox(sbox: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Calculate `sbox` forward and inverse transformations for 16-bit integers.

    Args:
        sbox ((16, ) ndarray[uint16]):
            Sbox for which to calculate transformations.

    Returns:
        tuple[ndarray, ndarray]:
            - (65536, ) ndarray[uint16]:
                Calculated forward sbox transformations.
            - (65536, ) ndarray[uint16]:
                Calculated inverse sbox transformations.
    """
    return (
        _sbox_transformation(sbox=sbox),
        _sbox_transformation(_get_inverse_sbox(sbox=sbox)),
    )


def _get_inverse_sbox(sbox: ndarray) -> ndarray:
    """
    Calculate inverse Sbox transformation.

    Args:
        sbox ((16, ) ndarray[uint16]):
            Sbox transformation for which to calculate inverse transformation.

    Returns:
        (16, ) ndarray[uint16]:
            Inverse transformation of the given `sbox`.
    """
    elements = arange(start=0, stop=16, dtype="uint16")
    transformed = sbox[elements]
    return array([
        int(where(transformed == element)[0])
        for element in elements
    ], dtype="uint16")


def _sbox_transformation(sbox: ndarray) -> ndarray:
    """
    Calculate table of `sbox` transformations for 16-bit integers.

    Args:
        sbox ((16, ) ndarray[uint16]):
            Sbox for which to calculate transformations.

    Returns:
        (65536, ) ndarray[uint16]:
            Calculated transformations.
    """
    sbox_mask = array([0b1111], dtype="uint16")
    elements = arange(start=0, stop=1 << 16, dtype="uint16")

    for fragment_id in range(_S_BOX_FRAGMENT):
        sbox_output = (
            sbox
            [(elements >> (_S_BOX_FRAGMENT * fragment_id)) & sbox_mask]
        )
        elements &= ~(sbox_mask << (_S_BOX_FRAGMENT * fragment_id))
        elements |= sbox_output << (_S_BOX_FRAGMENT * fragment_id)

    return elements
