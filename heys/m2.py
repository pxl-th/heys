from multiprocessing import (
    Manager,
    Process,
)
from typing import (
    Dict,
    List,
)

from numpy import (
    arange,
    argsort,
    array,
    einsum,
    ndarray,
    where,
    zeros_like,
)

from heys.cipher import Heys
from heys.utilities import (
    bitwise_dot,
    calculate_hamming_weight,
)


def m2(
    heys: Heys,
    inputs: ndarray,
    outputs: ndarray,
    approximations: Dict[int, Dict[int, float]],
    processes_number: int = 1,
    top_keys: int = 100,
) -> ndarray:
    print("Finding key candidates...")

    hamming = calculate_hamming_weight(bits=16)
    keys = []
    alpha_id = 0
    for alpha, betas in approximations.items():
        alpha_id += 1
        print(
            f"Processing alpha {alpha}: "
            f"{alpha_id}/{len(approximations.keys())}"
        )

        manager = Manager()
        keys_list = manager.list()

        betas_array = array(list(betas.keys()), dtype="uint16")
        betas_split = betas_array.shape[0] // processes_number
        processes = [
            Process(
                target=key_search_process,
                kwargs={
                    "output_keys": keys_list,
                    "heys": heys,
                    "alpha": alpha,
                    "betas": betas_array[pid * betas_split:
                                         (pid + 1) * betas_split],
                    "inputs": inputs,
                    "outputs": outputs,
                    "hamming": hamming,
                    "top_keys": top_keys,
                },
            ) for pid in range(processes_number)
        ]

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        keys.extend(keys_list)
    return array(keys, dtype="uint16")


def key_search_process(
    output_keys: List[int],
    heys: Heys,
    alpha: int,
    betas: ndarray,
    inputs: ndarray,
    outputs: ndarray,
    hamming: ndarray,
    top_keys: int = 100,
) -> None:
    key_candidates = arange(start=0, stop=1 << 16, dtype="uint16")
    possible_keys = zeros_like(key_candidates, dtype="uint64")

    for beta_id, beta in enumerate(betas):
        print(f"Processing beta {beta}: {beta_id + 1}/{betas.shape[0]}")
        for key in key_candidates:
            correlation = bitwise_dot(
                x=alpha,
                y=heys.permutation[heys.sbox[inputs ^ key]],
                hamming_weight=hamming,
            )
            correlation ^= bitwise_dot(
                x=beta,
                y=outputs,
                hamming_weight=hamming,
            )
            possible_keys[key] = abs(einsum(
                "e->",
                where(correlation == 0, 1, -1),
            ))
        output_keys.extend(argsort(possible_keys)[-top_keys:])
