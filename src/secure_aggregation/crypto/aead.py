import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


@dataclass
class AeadCiphertext:
    iv: bytes
    ciphertext: bytes
    tag: bytes


def _validate_key(key: bytes) -> None:
    if len(key) not in (16, 24, 32):
        raise ValueError("AES-GCM key must be 128/192/256 bits")


def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes | None = None, iv: bytes | None = None) -> AeadCiphertext:
    _validate_key(key)
    iv = iv or os.urandom(12)
    aesgcm = AESGCM(key)
    combined = aesgcm.encrypt(iv, plaintext, aad or b"")
    return AeadCiphertext(iv=iv, ciphertext=combined[:-16], tag=combined[-16:])


def aes_gcm_decrypt(key: bytes, iv: bytes, ciphertext: bytes, tag: bytes, aad: bytes | None = None) -> bytes:
    _validate_key(key)
    aesgcm = AESGCM(key)
    combined = ciphertext + tag
    return aesgcm.decrypt(iv, combined, aad or b"")
