from numpy import (
    arange,
    argmax,
    array,
    einsum,
    ndarray,
    where,
)

from heys.cipher import Heys
from heys.s_box import S_BOX
from heys.utilities import (
    bitwise_dot,
    calculate_hamming_weight,
)


# todo: swap bytes
def m2(heys: Heys, data: ndarray, approximations: ndarray) -> ndarray:
    inputs, outputs = data
    key_candidates = arange(start=1, stop=1 << 16, dtype="uint16")
    hamming = calculate_hamming_weight(bits=16)

    heys_round = (
        lambda elements, key:
        heys.permutation[heys.sbox[elements ^ key]]
    )

    keys = []
    for alpha, beta in approximations:
        possible_keys = []
        for key in key_candidates:
            corellation = bitwise_dot(
                x=alpha,
                y=heys_round(inputs, key),
                hamming_weight=hamming,
            )
            corellation ^= bitwise_dot(
                x=beta,
                y=outputs,
                hamming_weight=hamming,
            )
            possible_keys.append(
                einsum("e->", where(corellation == 0, 1, -1))
            )
        keys.append(argmax(possible_keys))

    return array(keys)
