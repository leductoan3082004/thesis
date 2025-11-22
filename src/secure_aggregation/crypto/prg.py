from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def _derive_key_iv(seed: bytes) -> tuple[bytes, bytes]:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=48, salt=None, info=b"secure-agg/prg")
    material = hkdf.derive(seed)
    return material[:32], material[32:]


def prg_bytes(seed: bytes, length: int) -> bytes:
    """
    Deterministic PRG based on AES-CTR. For a given seed and length, output is stable.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return b""
    key, iv = _derive_key_iv(seed)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
    encryptor = cipher.encryptor()
    return encryptor.update(b"\x00" * length) + encryptor.finalize()
