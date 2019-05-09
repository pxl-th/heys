from pytest import mark

from numpy import array

from heys.cipher import Heys
from heys.s_block import S_BOX


def test_permutation_fixed_point():
    """
    Every 5th bit of a 16-bit number should remain fixed after permutation.
    """
    hayes = Heys(s_block_table=S_BOX, keys=array([0], dtype="uint16"))
    fixed_input = 0b1000010000100001
    assert fixed_input == hayes.permutation[fixed_input]


@mark.parametrize(
    "input_number, target_number",
    [
        (0b0100000011010110, 0b0010101100010010),
        (0b0111101000011101, 0b0101100111001011),
        (0b1001, 0b0001000000000001),
    ],
)
def test_permutation(input_number, target_number):
    """
    Every ith bit of a jth block goes to jth bit of a ith block.
    """
    hayes = Heys(s_block_table=S_BOX, keys=array([0], dtype="uint16"))
    assert input_number == hayes.permutation[target_number]


def test_reverse_permutation():
    """
    Applying permutation second time, should return original input number.
    """
    hayes = Heys(s_block_table=S_BOX, keys=array([0], dtype="uint16"))
    for input_number in range(1 << 16):
        target_number = hayes.permutation[hayes.permutation[input_number]]
        assert input_number == target_number
