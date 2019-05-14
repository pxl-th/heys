from multiprocessing import (
    Manager,
    Process,
)
from typing import (
    Dict,
    List,
    Tuple,
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

__all__ = ["m2"]


def m2(
    heys: Heys,
    data: Tuple[ndarray, ndarray],
    approximations: Dict[int, Dict[int, float]],
    processes_number: int = 1,
    top_keys: int = 100,
) -> ndarray:
    """
    M2 algorithm for finding key-candidates,
    given linear approximations and
    corresponding pairs of open-text and ciphertext.

    Args:
        heys (:class:`Heys`):
            Heys cipher from which round function will be taken.
        data (tuple[(M, ) ndarray[uint16], (M, ) ndarray[uint16]]):
            Tuple where the first element is the array opentexts
            and the second element is the array of ciphertexts,
            corresponding to the opentexts.
        approximations (dict[int, dict[int, float]]):
            Linear approximations.
        processes_number (int):
            Number of processors to use.
        top_keys (int):
            Number of key-candidates to select in :meth:`m2` algorithm
            that have highest statistics.

    Returns:
        (M, ) ndarray[uint16]:
            Array of key candidates, might contain duplicates of keys.
    """
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
                target=_key_search_process,
                kwargs={
                    "output_keys": keys_list,
                    "heys": heys,
                    "alpha": alpha,
                    "betas": betas_array[pid * betas_split:
                                         (pid + 1) * betas_split],
                    "data": data,
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


def _key_search_process(
    output_keys: List[int],
    heys: Heys,
    alpha: int,
    betas: ndarray,
    data: Tuple[ndarray, ndarray],
    hamming: ndarray,
    top_keys: int = 100,
) -> None:
    key_candidates = arange(start=0, stop=1 << 16, dtype="uint16")
    possible_keys = zeros_like(key_candidates, dtype="uint64")

    for beta_id, beta in enumerate(betas):
        print(f"| Processing beta {beta}: {beta_id + 1}/{betas.shape[0]}")
        for key in key_candidates:
            correlation = bitwise_dot(
                x=alpha,
                y=heys.permutation[heys.sbox[data[0] ^ key]],
                hamming_weight=hamming,
            )
            correlation ^= bitwise_dot(
                x=beta,
                y=data[1],
                hamming_weight=hamming,
            )
            possible_keys[key] = abs(einsum(
                "e->",
                where(correlation == 0, 1, -1),
            ))
        output_keys.extend(argsort(possible_keys)[-top_keys:])
