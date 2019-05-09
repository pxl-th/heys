from pytest import mark

from heys.utilities import (
    bitwise_dot,
    calculate_hamming_weight,
)


HAMMING_TABLE = calculate_hamming_weight(bits=16)


@mark.parametrize(
    "x, y, target",
    [
        (0, 0, 0),
        (0b11110000, 0b00001111, 0),
        (0b11110000, 0b00011111, 1),
        (0b1010000010100000, 0b1010000010100000, 0),
        (0b1111111111111111, 0b1111111111111110, 1),
    ],
)
def test_bitwise_dot(x, y, target):
    assert target == int(bitwise_dot(x=x, y=y, hamming_weight=HAMMING_TABLE))
