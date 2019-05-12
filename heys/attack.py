from collections import defaultdict
from os.path import isfile
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


def attack(
    alphas: ndarray,
    alphas_attempts: int,
    approximations_file: str,
    probability_threshold: float = 1e-8,
):
    heys_keys = array(
        [0x42, 0xfc, 0xaf, 0x13, 0x1488, 0x1984, 0xeaa],
        dtype="uint16",
    )
    heys = Heys(sbox=S_BOX, keys=heys_keys)

    approximations = calculate_approximations(
        heys=heys,
        alphas=alphas,
        alphas_attempts=alphas_attempts,
        approximations_file=approximations_file,
        approximations_number=600,
        probability_threshold=probability_threshold,
    )

    inputs = arange(start=0, stop=5000, dtype="uint16")
    outputs = heys.encrypt(message=inputs)

    key_candidates = m2(
        heys=heys,
        inputs=inputs,
        outputs=outputs,
        approximations=approximations,
    )
    print(key_candidates.shape)
    print(key_candidates)


def calculate_approximations(
    heys: Heys,
    alphas: ndarray,
    alphas_attempts: int,
    approximations_file: str,
    approximations_number: int,
    probability_threshold: float = 1e-8,
) -> dict:
    approximations = defaultdict(lambda: dict())
    if isfile(approximations_file):
        with open(approximations_file, "rb") as saved_file:
            approximations = load(saved_file)

    total_approximations = sum([
        len(alpha_approximations)
        for _, alpha_approximations in approximations.items()
    ])
    print(f"Total approximations {total_approximations} loaded.")

    if total_approximations >= approximations_number:
        return approximations

    print(f"Finding linear approximations for the Heys cipher...")
    for alpha_id, alpha in enumerate(alphas):
        print(
            f"Finding approximations for {alpha_id}/{alphas.shape[0]} alpha..."
        )

        for alpha_attempt in range(alphas_attempts):
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
                dump(approximations, save_file)

    return approximations


def main():
    alphas = array([
        0b1111,
        0b11110000,
        0b111100000000,
        0b1111000000000000,
    ], dtype="uint16")
    attack(
        alphas=alphas,
        alphas_attempts=5,
        approximations_file="heys-approximations.pkl",
    )


if __name__ == "__main__":
    main()
