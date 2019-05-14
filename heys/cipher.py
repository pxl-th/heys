from os.path import (
    abspath,
    isfile,
    join,
)
from pickle import (
    dump,
    load,
)

from numpy import (
    arange,
    ndarray,
    zeros_like,
)

from heys.s_box import calculate_sbox

__all__ = ["Heys"]


class Heys:
    FRAGMENT = 4

    PERMUTATION_FILE = "heys-permutation.pkl"
    SBOX_FILE = "heys-sbox.pkl"
    SBOX_INVERSE_FILE = "heys-sbox-inverse.pkl"

    def __init__(self, sbox: ndarray, keys: ndarray):
        self._keys = keys.copy()
        self._rounds = self._keys.shape[0] - 1

        self._sbox_fragment = sbox

        self._permutation = None
        self._sbox = None
        self._sbox_inverse = None

        self._load_cache()

    @property
    def keys(self) -> ndarray:
        return self._keys

    @property
    def rounds(self) -> int:
        return self._rounds

    @property
    def permutation(self) -> ndarray:
        return self._permutation

    @property
    def sbox_fragment(self) -> ndarray:
        return self._sbox_fragment

    @property
    def sbox(self) -> ndarray:
        return self._sbox

    @property
    def sbox_inverse(self) -> ndarray:
        return self._sbox_inverse

    def _load_cache(self):
        file_directory_path = abspath(join(__file__, ".."))
        permutation_path = join(file_directory_path, Heys.PERMUTATION_FILE)
        sbox_path = join(file_directory_path, Heys.SBOX_FILE)
        sbox_inverse_path = join(file_directory_path, Heys.SBOX_INVERSE_FILE)

        cache_intact = (
            isfile(permutation_path)
            and isfile(sbox_path)
            and isfile(sbox_inverse_path)
        )
        if cache_intact:
            with open(sbox_path, "rb") as sbox_file:
                self._sbox = load(sbox_file)
            with open(sbox_inverse_path, "rb") as sbox_inverse_file:
                self._sbox_inverse = load(sbox_inverse_file)
            with open(permutation_path, "rb") as permutation_file:
                self._permutation = load(permutation_file)
        else:
            self._create_cache(
                permutation_path=permutation_path,
                sbox_path=sbox_path,
                sbox_inverse_path=sbox_inverse_path,
            )

    def _create_cache(
        self,
        permutation_path: str,
        sbox_path: str,
        sbox_inverse_path: str,
    ):
        self._permutation = Heys._calculate_permutation()
        self._sbox, self._sbox_inverse = calculate_sbox(
            sbox=self._sbox_fragment,
        )

        with open(permutation_path, "wb") as permutation_file:
            dump(obj=self._permutation, file=permutation_file)
        with open(sbox_path, "wb") as sbox_file:
            dump(obj=self._sbox, file=sbox_file)
        with open(sbox_inverse_path, "wb") as sbox_inverse_file:
            dump(obj=self._sbox_inverse, file=sbox_inverse_file)

    def encrypt(self, message: ndarray) -> ndarray:
        output = message.copy()

        for round_id in range(self._rounds):
            output ^= self._keys[round_id]
            output = self._sbox[output]
            output = self._permutation[output]

        output ^= self._keys[-1]
        return output

    def decrypt(self, ciphertext: ndarray) -> ndarray:
        output = ciphertext.copy()

        output ^= self._keys[-1]
        for round_id in reversed(range(self._rounds)):
            output = self._permutation[output]
            output = self._sbox_inverse[output]
            output ^= self._keys[round_id]

        return output

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
