"""
Branch and Bound algorithm for finding linear approximations.
"""
from collections import defaultdict
from typing import Dict

from numpy import (
    arange,
    ndarray,
    ones_like,
    zeros,
)

from heys.cipher import Heys
from heys.utilities import (
    calculate_hamming_weight,
    linear_potential,
)

__all__ = ["branch_bound"]


def branch_bound(
    heys: Heys,
    alpha: int,
    probability_threshold: float,
) -> Dict[int, float]:
    sbox_probabilities = _sbox_linear_approximations(s_box=heys.sbox_fragment)
    betas = arange(start=1, stop=1 << 16, dtype="uint16")

    previous_round = {alpha: 1}
    for round_id in range(heys.rounds - 1):
        print(
            f"Round: {round_id + 1} | Branches: {len(previous_round.keys())}"
        )

        current_round = defaultdict(lambda: 0)
        for previous_element, previous_probability in previous_round.items():
            current_elements = zip(
                betas,
                _approximation_probability(
                    input_element=previous_element,
                    output_elements=heys.permutation[betas],
                    sbox_probabilities=sbox_probabilities,
                ),
            )
            for current_element, current_probability in current_elements:
                current_round[current_element] += (
                    previous_probability
                    * current_probability
                )
        previous_round = {
            element: probability
            for element, probability in current_round.items()
            if probability > probability_threshold
        }
    return previous_round


def _approximation_probability(
    input_element: int,
    output_elements: ndarray,
    sbox_probabilities: ndarray,
) -> ndarray:
    mask = 0b1111
    probabilities = ones_like(output_elements, dtype="float64")
    for fragment_id in range(4):
        probabilities *= (
            sbox_probabilities
            [(input_element >> (4 * fragment_id)) & mask]
            [(output_elements >> (4 * fragment_id)) & mask]
        )
    return probabilities


def _sbox_linear_approximations(s_box: ndarray) -> ndarray:
    numbers = 1 << 4
    approximations = zeros((16, 16), dtype="float64")
    input_numbers = arange(numbers, dtype="uint16")
    hamming = calculate_hamming_weight(bits=4)

    for alpha in range(numbers):
        for beta in range(numbers):
            approximations[alpha, beta] = (linear_potential(
                inputs=input_numbers,
                alpha=alpha,
                beta=beta,
                transformation=lambda x: s_box[x],
                hamming_weight=hamming,
            ))
    return approximations
