from numpy import (
    arange,
    where,
)
from numpy.random import randint
from numpy.testing import assert_allclose

from hayes.cipher import Hayes
from hayes.s_block import S_BOX
from hayes.utilities import (
    calculate_hamming_weight,
    linear_potential,
)


def test_linear_potential_property():
    """
    For linear potential holds LP(a, 0) = [a == 0].
    """
    bits_number = 16
    inputs_number = 1 << bits_number
    alphas_batch_size = 128

    hamming = calculate_hamming_weight(bits=bits_number)
    keys = randint(low=inputs_number, size=7, dtype="uint16")
    keys_batch = keys[0].reshape((1, -1))
    hayes = Hayes(s_block_table=S_BOX, keys=keys)
    hayes_round = (
        lambda elements:
        hayes.permutation[
            hayes.s_block[elements ^ hayes.keys[0]],
        ]
    )

    inputs = arange(
        start=0,
        stop=inputs_number,
        dtype="uint16",
    ).reshape((1, -1))
    alphas_batch = randint(
        low=inputs_number,
        size=(alphas_batch_size, 1),
        dtype="uint16",
    )

    potential = linear_potential(
        inputs=inputs,
        alpha=alphas_batch,
        beta=0,
        keys=keys_batch,
        hayes=hayes,
        hamming_weight=hamming,
    ).reshape((alphas_batch_size, 1))

    assert_allclose(where(alphas_batch == 0, 1, 0), potential)
