from collections import Counter
from os.path import isfile
from pickle import (
    dump,
    load,
)
from typing import (
    Dict,
    Tuple,
)

from numpy import (
    arange,
    array,
    fromfile,
    ndarray,
)

from heys.branch_bound import branch_bound
from heys.cipher import Heys
from heys.m2 import m2
from heys.s_box import S_BOX

__all__ = [
    "attack",
    "calculate_approximations",
]


def attack(
    heys: Heys,
    data: Tuple[ndarray, ndarray],
    approximations: Dict[int, Dict[int, float]],
    max_approximations_number: int,
    probability_threshold: float,
    processes_number: int,
    top_keys: int,
) -> Tuple[Dict[int, Dict[int, float]], Dict[int, int]]:
    r"""
    Perform an attack on first round-key of SPN block cipher.

    Args:
        heys (:class:`Heys`):
            Instance of :class:`Heys` from which round functions
            will be taken.
        data (tuple[(M, ) ndarray[uint16], (M, ) ndarray[uint16]]):
            Tuple where the first element is the array opentexts
            and the second element is the array of ciphertexts,
            corresponding to the opentexts.
        approximations (dict[int, dict[int, float]]):
            - int:
                Initial `alpha` value.
            - dict[int, float]:
                If dictionaries has less than `max_approximations_number`
                approximations, then algorithm will try to find more
                approximations using :meth:`branch_bound` algorithm.

                - int:
                    Founded `beta` value for the initial `alpha` value.
                - float:
                    Probability :math:`p` of linear
                    approximation (`alpha`, `beta`),
                    for which holds :math:`p \ge p^*`,
                    where :math:`p^*` is the `probability_threshold`.
        max_approximations_number (int):
            Amount of approximations that :meth:`branch_bounds` algorithm
            should find before stopping.
            **Note**, if for the given `probability_threshold`
            algorithm found less than `max_approximations_number`
            number of approximations, then you should
            decrease `probability_threshold` and start it again.
        probability_threshold (float):
            Probability threshold which determines,
            what approximations will be selected.
            The higher the value the faster (but more inaccurate)
            algorithm is, and vise-versa, the lower the value,
            the more robust it is (but will require much more computations).
        processes_number (int):
            Number of processors to use in :meth:`m2` algorithm.
        top_keys (int):
            Number of key-candidates to select in :meth:`m2` algorithm
            that have highest statistics.

    Returns:
        dict[int, int]:
            - int:
                Key candidate.
            - int:
                Number of times, key candidate got in `top_keys` list.
                The higher the number, the higher the chances of it
                being a real key.
    """
    approximations = calculate_approximations(
        heys=heys,
        approximations=approximations,
        max_approximations_number=max_approximations_number,
        probability_threshold=probability_threshold,
    )
    key_candidates = m2(
        heys=heys,
        data=data,
        approximations=approximations,
        processes_number=processes_number,
        top_keys=top_keys,
    )
    return approximations, dict(Counter(key_candidates))


def calculate_approximations(
    heys: Heys,
    approximations: Dict[int, Dict[int, float]],
    max_approximations_number: int,
    probability_threshold: float,
) -> Dict[int, Dict[int, float]]:
    r"""
    Find linear approximations for the SPN
    using :meth:`branch_bound` algorithm.
    If `heys` has :math:`r` rounds, then founded approximations are for
    :math:`r - 1` rounds, because we attack first round key.
    Linear approximation is a pair :math:`(\alpha, \beta)`, such that

    .. math::
        \forall X, Y: \alpha \cdot X \oplus \beta \cdot Y = 0.

    Args:
        heys (Heys):
            Cipher from which `sbox` and number of `rounds` is taken.
        approximations (dict[int, dict[int, float]]):
            - int:
                Initial `alpha` value.
            - dict[int, float]:
                If dictionaries has less than `max_approximations_number`
                approximations, then algorithm will try to find more
                approximations using :meth:`branch_bound` algorithm.

                - int:
                    Founded `beta` value for the initial `alpha` value.
                - float:
                    Probability :math:`p` of linear
                    approximation (`alpha`, `beta`),
                    for which holds :math:`p \ge p^*`,
                    where :math:`p^*` is the `probability_threshold`.
        max_approximations_number (int):
            Amount of approximations that :meth:`branch_bounds` algorithm
            should find before stopping.
            **Note**, if for the given `probability_threshold`
            algorithm found less than `max_approximations_number`
            number of approximations, then you should
            decrease `probability_threshold` and start it again.
        probability_threshold (float):
            Probability threshold which determines,
            what approximations will be selected.
            The higher the value the faster (but more inaccurate)
            algorithm is, and vise-versa, the lower the value,
            the more robust it is (but will require much more computations).

    Returns:
        dict[int, dict[int, float]]:
            - int:
                Initial `alpha` value.
            - dict[int, float]:
                - int:
                    Founded `beta` value for the initial `alpha` value.
                - float:
                    Probability :math:`p` of linear
                    approximation (`alpha`, `beta`),
                    for which holds :math:`p \ge p^*`,
                    where :math:`p^*` is the `probability_threshold`.
    """
    approximations_amount = (lambda holder: sum(
        len(alpha_approximations)
        for alpha_approximations in holder.values()
    ))

    total_approximations = approximations_amount(approximations)
    print(f"Total approximations {total_approximations}.")
    if total_approximations >= max_approximations_number:
        return dict(approximations)

    print(f"Finding linear approximations for the Heys cipher...")
    for alpha_id, alpha in enumerate(approximations.keys()):
        print(
            "Finding approximations for "
            f"{alpha_id + 1}/{len(approximations.keys())} alpha..."
        )

        approximations[alpha].update(branch_bound(
            heys=heys,
            alpha=alpha,
            probability_threshold=probability_threshold,
        ))
        total_approximations = approximations_amount(approximations)

        print(f"Total approximations {total_approximations}.")
        if total_approximations >= max_approximations_number:
            break

    return dict(approximations)


def main():
    # keys_file = r"C:\Users\tonys\Downloads\heys\heys_keys"
    # input_file = r"C:\Users\tonys\Downloads\heys\heys_input"
    output_file = r"C:\Users\tonys\Downloads\heys\heys_output"

    approximations_file = "heys-approximations-full.pkl"
    key_candidates_file = "heys-keys.pkl"

    heys_keys = array(
        [0xafaf, 0x1234, 0x9998, 0x3457, 0x12, 0x1488, 0x7376],
        dtype="uint16",
    )
    heys = Heys(sbox=S_BOX, keys=heys_keys)
    opentexts = arange(start=0, stop=20000, dtype="uint16")
    ciphertexts = fromfile(file=output_file, dtype="uint16")

    # opentexts.tofile(input_file)
    # heys_keys.tofile(keys_file)

    approximations = {
        0b1111000000000000: {},
        0b111100000000: {},
        0b11110000: {},
        0b1111: {},
    }

    if isfile(approximations_file):
        with open(approximations_file, "rb") as saved_file:
            approximations = load(saved_file)

    approximations, key_candidates = attack(
        heys=heys,
        data=(opentexts, ciphertexts),
        approximations=approximations,
        max_approximations_number=200,
        probability_threshold=5e-5,
        processes_number=6,
        top_keys=100,
    )

    with open(approximations_file, "wb") as save_file:
        dump(dict(approximations), save_file)
    with open(key_candidates_file, "wb") as save_file:
        dump(dict(key_candidates), save_file)

    print(list(reversed(sorted(
        key_candidates.items(),
        key=lambda element: element[1],
    ))))


def print_key_candidates(key_file: str):
    with open(key_file, "rb") as key:
        key_candidates = load(key)

    print([
        [hex(candidate), counter]
        for candidate, counter in reversed(sorted(
            key_candidates.items(),
            key=lambda element: element[1],
        ))
    ])


def print_approximations(approximations_file: str):
    with open(approximations_file, "rb") as appr_file:
        approximations = load(appr_file)

    for alpha, betas in approximations.items():
        hex_betas = {
            hex(beta): prob
            for beta, prob in betas.items()
        }
        print(f"{hex(alpha)} | {hex_betas}")


if __name__ == "__main__":
    # main()
    # print_key_candidates(key_file="heys-keys.pkl")
    print_approximations(approximations_file="heys-approximations-full.pkl")
