from collections import (
    Counter,
    defaultdict,
)
from os.path import isfile
from typing import Dict
from pickle import (
    dump,
    load,
)

from numpy import (
    arange,
    array,
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
        alphas: ndarray,
        approximations_file: str,
        keys_file: str,
        approximations_number: int = 300,
        probability_threshold: float = 1e-5,
        processes_number: int = 1,
        top_keys: int = 100,
):
    """
    Perform an attack on first round-key of SPN block cipher.

    Args:
        alphas ((N, )ndarray[uint16]):
            Initial values for `branch-and-bounds` algorithm.
        approximations_file (str):
            Filepath, where founded approximations will be saved.
        keys_file (str):
            File, where to save founded key-candidates.
        approximations_number (int):
            Number of approximations, that algorithm should find
            to stop.
            If for the given `probability_threshold` algorithm found
            less than `approximations_number` number of approximations,
            then you should decrease `probability_threshold`
            and start again.
        probability_threshold (float):
            Probability threshold which determines,
            what approximations will be selected.
            The higher the value the faster (but more inaccurate)
            algorithm is, and vise-versa, the lower the value,
            the more robust it is, but will require much more computations.
        processes_number (int):
            Number of processors to use in :meth:`m2` algorithm.
        top_keys (int):
            Number of key-candidates to select in :meth:`m2` algorithm
            that have highest statistics.
    """
    heys_keys = array(
        [0xff2c, 0x1488, 0xa23f, 0xe323, 0x1444, 0x2012, 0xeaa],
        dtype="uint16",
    )
    heys = Heys(sbox=S_BOX, keys=heys_keys)
    inputs = arange(start=0, stop=3000, dtype="uint16")
    ciphertexts = heys.encrypt(message=inputs)

    approximations = calculate_approximations(
        heys=heys,
        alphas=alphas,
        approximations_file=approximations_file,
        approximations_number=approximations_number,
        probability_threshold=probability_threshold,
    )

    key_candidates = m2(
        heys=heys,
        inputs=inputs,
        ciphertexts=ciphertexts,
        approximations=approximations,
        processes_number=processes_number,
        top_keys=top_keys,
    )

    keys_counter = Counter(key_candidates)
    print(keys_counter)
    with open(keys_file, "wb") as f:
        dump(keys_counter, f)


def calculate_approximations(
        heys: Heys,
        alphas: ndarray,
        approximations_file: str,
        approximations_number: int,
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
        alphas ((N, )ndarray[uint16]):
            Initial values for `branch-and-bounds` algorithm.
        approximations_file (str):
            Filepath, where founded approximations will be saved.
        approximations_number (int):
            Number of approximations, that algorithm should find
            to stop.
            If for the given `probability_threshold` algorithm found
            less than `approximations_number` number of approximations,
            then you should decrease `probability_threshold`
            and start again.
        probability_threshold (float):
            Probability threshold which determines,
            what approximations will be selected.
            The higher the value the faster (but more inaccurate)
            algorithm is, and vise-versa, the lower the value,
            the more robust it is, but will require much more computations.

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
    approximations: Dict[int, Dict[int, float]] = defaultdict(lambda: dict())
    if isfile(approximations_file):
        with open(approximations_file, "rb") as saved_file:
            approximations.update(load(saved_file))

    total_approximations = sum([
        len(alpha_approximations)
        for _, alpha_approximations in approximations.items()
    ])
    print(f"Total approximations {total_approximations} loaded.")

    if total_approximations >= approximations_number:
        return dict(approximations)

    print(f"Finding linear approximations for the Heys cipher...")
    for alpha_id, alpha in enumerate(alphas):
        print(
            "Finding approximations for "
            f"{alpha_id + 1}/{alphas.shape[0]} alpha..."
        )
        approximations[alpha].update(branch_bound(
            heys=heys,
            alpha=alpha,
            probability_threshold=probability_threshold,
        ))
        total_approximations = sum([
            len(alpha_approximations)
            for _, alpha_approximations in approximations.items()
        ])
        print(f"Total approximations {total_approximations}.")
        with open(approximations_file, "wb") as save_file:
            dump(dict(approximations), save_file)

        if total_approximations >= approximations_number:
            break

    return dict(approximations)


def main():
    alphas = array([
        0b1111000000000000,
        0b111100000000,
        0b11110000,
        0b1111,
    ], dtype="uint16")
    attack(
        alphas=alphas,
        approximations_file="heys-approximations-full.pkl",
        keys_file="heys-keys.pkl",
        approximations_number=100,
        probability_threshold=5e-5,
        processes_number=6,
        top_keys=200,
    )


if __name__ == "__main__":
    main()
