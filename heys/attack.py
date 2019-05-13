from collections import (
    Counter,
    defaultdict,
)
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
    approximations_number: int = 600,
    probability_threshold: float = 1e-8,
    processes_number: int = 1,
):
    heys_keys = array(
        [0x42, 0xfc, 0xaf, 0x13, 0x1488, 0x1984, 0xeaa],
        dtype="uint16",
    )
    heys = Heys(sbox=S_BOX, keys=heys_keys)
    inputs = arange(start=0, stop=30000, dtype="uint16")
    outputs = heys.encrypt(message=inputs)

    approximations = calculate_approximations(
        heys=heys,
        alphas=alphas,
        alphas_attempts=alphas_attempts,
        approximations_file=approximations_file,
        approximations_number=approximations_number,
        probability_threshold=probability_threshold,
    )

    key_candidates = m2(
        heys=heys,
        inputs=inputs,
        outputs=outputs,
        approximations=approximations,
        processes_number=processes_number,
    )
    count = Counter(key_candidates)
    print(count)
    print(key_candidates.shape)
    print(key_candidates)
    with open("kc.pkl", "wb") as f:
        dump(key_candidates, f)
    with open("kcc.pkl", "wb") as f:
        dump(count, f)


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
            "Finding approximations for "
            f"{alpha_id}/{alphas.shape[0]} alpha..."
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
                dump(dict(approximations), save_file)

            if total_approximations >= approximations_number:
                return dict(approximations)

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
        alphas_attempts=5,
        approximations_file="heys-approximations.pkl",
        approximations_number=300,
        probability_threshold=2e-8,
        processes_number=6,
    )


if __name__ == "__main__":
    main()
