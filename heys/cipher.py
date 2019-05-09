from os.path import (
    abspath,
    isfile,
    join,
)
from pickle import (
    dump,
    load,
)
from typing import Tuple

from numpy import (
    arange,
    array,
    full,
    ndarray,
    where,
    zeros_like,
)

__all__ = ["Heys"]


class Heys:
    FRAGMENT = 4

    PERMUTATION_FILE = "heys-permutation.pkl"
    S_BLOCK_FILE = "heys-s-block.pkl"
    S_BLOCK_INVERSE_FILE = "heys-s-block-inverse.pkl"

    def __init__(self, s_block_table: ndarray, keys: ndarray):
        self._keys = keys.copy().byteswap(inplace=True)
        self._rounds = self._keys.shape[0] - 1

        self._s_block_mask = full(shape=1, fill_value=0b1111, dtype="uint16")
        self._s_block_substitution = s_block_table
        self._s_block_inverse_substitution = self._calculate_s_block_inverse()

        self._permutation = None
        self._s_block = None
        self._s_block_inverse = None

        self._load_cache()

    def _load_cache(self):
        file_directory_path = abspath(join(__file__, ".."))
        permutation_path = join(file_directory_path, Heys.PERMUTATION_FILE)
        s_block_path = join(file_directory_path, Heys.S_BLOCK_FILE)
        s_block_inverse_path = join(
            file_directory_path,
            Heys.S_BLOCK_INVERSE_FILE,
        )

        cache_intact = (
            isfile(permutation_path)
            and isfile(s_block_path)
            and isfile(s_block_inverse_path)
        )
        if cache_intact:
            with open(s_block_path, "rb") as s_block_file:
                self._s_block = load(s_block_file)
            with open(s_block_inverse_path, "rb") as s_block_inverse_file:
                self._s_block_inverse = load(s_block_inverse_file)
            with open(permutation_path, "rb") as permutation_file:
                self._permutation = load(permutation_file)
        else:
            self._create_cache(
                permutation_path=permutation_path,
                s_block_path=s_block_path,
                s_block_inverse_path=s_block_inverse_path,
            )

    def _create_cache(
        self,
        permutation_path: str,
        s_block_path: str,
        s_block_inverse_path: str,
    ):
        self._permutation = Heys._calculate_permutation()
        self._s_block, self._s_block_inverse = self._calculate_s_block()

        with open(permutation_path, "wb") as permutation_file:
            dump(obj=self._permutation, file=permutation_file)
        with open(s_block_path, "wb") as s_block_file:
            dump(obj=self._s_block, file=s_block_file)
        with open(s_block_inverse_path, "wb") as s_block_inverse_file:
            dump(obj=self._s_block_inverse, file=s_block_inverse_file)

    @property
    def keys(self) -> ndarray:
        return self._keys

    @property
    def permutation(self) -> ndarray:
        return self._permutation

    @property
    def s_block(self) -> ndarray:
        return self._s_block

    @property
    def s_block_inverse(self) -> ndarray:
        return self._s_block_inverse

    def encrypt(self, message: ndarray) -> ndarray:
        output = message.copy().byteswap(inplace=True)

        for round_id in range(self._rounds):
            output ^= self._keys[round_id]
            output = self._s_block[output]
            output = self._permutation[output]

        output ^= self._keys[-1]
        return output

    def decrypt(self, ciphertext: ndarray) -> ndarray:
        output = ciphertext.copy()

        output ^= self._keys[-1]
        for round_id in reversed(range(self._rounds)):
            output = self._permutation[output]
            output = self._s_block_inverse[output]
            output ^= self._keys[round_id]

        return output.byteswap(inplace=True)

    def _s_block_raw(
        self,
        elements: ndarray,
        inverse: bool = False,
    ) -> ndarray:
        transformation = (
            self._s_block_substitution
            if not inverse else
            self._s_block_inverse_substitution
        )
        for fragment_id in range(Heys.FRAGMENT):
            s_block_output = transformation[
                (elements >> (Heys.FRAGMENT * fragment_id))
                & self._s_block_mask
                ]
            elements &= ~(self._s_block_mask << (Heys.FRAGMENT * fragment_id))
            elements |= s_block_output << (Heys.FRAGMENT * fragment_id)

        return elements

    def _calculate_s_block_inverse(self) -> ndarray:
        elements = array(list(range(16)), dtype="uint16")
        transformed = self._s_block_substitution[elements]
        reverse_s_block = array([
            int(where(transformed == element)[0])
            for element in elements
        ], dtype="uint16")
        return reverse_s_block

    @staticmethod
    def _calculate_permutation() -> ndarray:
        words = arange(start=0, stop=1 << 16, dtype="uint16")
        shuffled = zeros_like(words, dtype="uint16")

        for word_id, word in enumerate(words):
            for bit_id in range(16):
                block_id = bit_id // Heys.FRAGMENT
                bit_pos = bit_id % Heys.FRAGMENT
                tmp_bit = (word & (1 << bit_id)) >> bit_id
                shuffled[word_id] |= (
                    (tmp_bit << (bit_pos * Heys.FRAGMENT))
                    << block_id
                )

        return shuffled

    def _calculate_s_block(self) -> Tuple[ndarray, ndarray]:
        words = arange(start=0, stop=1 << 16, dtype="uint16")
        return (
            self._s_block_raw(elements=words),
            self._s_block_raw(elements=words, inverse=True),
        )