from numpy import (
    arange,
    einsum,
    ndarray,
    where,
    zeros,
)

from tqdm import tqdm

from hayes.cipher import Hayes
from hayes.s_block import S_BLOCK

__all__ = [
    "bitwise_dot",
    "calculate_hamming_weight",
    "average_linear_potential",
    "linear_potential",
]


def average_linear_potential(
    inputs: ndarray,
    alpha: int,
    beta: int,
    hayes: Hayes,
    hamming_weight: ndarray,
    keys_batch_size: int = 256,
) -> ndarray:
    numbers = 1 << 16
    keys_batches = numbers // keys_batch_size
    keys_batch = arange(keys_batch_size, dtype="uint16").reshape((-1, 1))
    potential_einsum, average_einsum = "rc->r", "e->"

    average_potential = 0
    for _ in tqdm(range(keys_batches)):
        potential = bitwise_dot(
            x=beta,
            y=hayes.permutation[hayes.s_block[inputs ^ keys_batch]],
            hamming_weight=hamming_weight,
        )
        potential ^= bitwise_dot(
            x=alpha,
            y=inputs,
            hamming_weight=hamming_weight,
        )
        potential = (
            (einsum(potential_einsum, where(potential == 0, 1, -1))
             / numbers) ** 2
        )
        average_potential += einsum(average_einsum, potential)
        keys_batch += keys_batch_size

    return average_potential / numbers


def linear_potential(
    inputs: ndarray,
    alpha: int,
    beta: int,
    keys: ndarray,
    hayes: Hayes,
    hamming_weight: ndarray,
) -> ndarray:
    potential = bitwise_dot(
        x=beta,
        y=hayes.permutation[hayes.s_block[inputs ^ keys]],
        hamming_weight=hamming_weight,
    )
    potential ^= bitwise_dot(x=alpha, y=inputs, hamming_weight=hamming_weight)
    return (einsum("rc->r", where(potential == 0, 1, -1)) / (1 << 16)) ** 2


def bitwise_dot(x, y, hamming_weight: ndarray):
    return hamming_weight[x & y] & 0b1


def calculate_hamming_weight(bits: int = 16) -> ndarray:
    weight_table = zeros(1 << bits, dtype="uint16")
    for number in range(1 << bits):
        weight_table[number] = bin(number).count("1")
    return weight_table


def main():
    bits_number = 16
    inputs_number = 1 << bits_number

    hamming = calculate_hamming_weight(bits=bits_number)
    hayes = Hayes(s_block_table=S_BLOCK, keys=zeros(6, dtype="uint16"))
    inputs = arange(inputs_number, dtype="uint16").reshape((1, -1))

    # total_potential = 0
    # for alpha in tqdm(range(inputs_number)):
    #     lp = linear_potential(
    #         inputs=inputs,
    #         alpha=alpha,
    #         beta=43342,
    #         keys=hayes.keys[0],
    #         hayes=hayes,
    #         hamming_weight=hamming,
    #     )
    #     total_potential += lp
    # print(f"Total potential {total_potential}")

    for beta in range(inputs_number):
        av = average_linear_potential(
            inputs=inputs,
            alpha=42422,
            beta=beta,
            hayes=hayes,
            hamming_weight=hamming,
        )
        if av > 0:
            print(av)


if __name__ == '__main__':
    main()
