import binascii

import pytest

from secure_aggregation.crypto import aes_gcm_decrypt, aes_gcm_encrypt


def test_aes_gcm_matches_known_vector() -> None:
    key = binascii.unhexlify("00000000000000000000000000000000")
    iv = binascii.unhexlify("000000000000000000000000")
    plaintext = binascii.unhexlify("00000000000000000000000000000000")
    expected_ciphertext = binascii.unhexlify("0388dace60b6a392f328c2b971b2fe78")
    expected_tag = binascii.unhexlify("ab6e47d42cec13bdf53a67b21257bddf")

    ct = aes_gcm_encrypt(key, plaintext, aad=None, iv=iv)
    assert ct.ciphertext == expected_ciphertext
    assert ct.tag == expected_tag
    recovered = aes_gcm_decrypt(key, iv, ct.ciphertext, ct.tag, aad=None)
    assert recovered == plaintext


def test_aes_gcm_rejects_modified_tag() -> None:
    key = b"\x00" * 16
    iv = b"\x01" * 12
    ct = aes_gcm_encrypt(key, b"secret", aad=b"context", iv=iv)
    bad_tag = b"\x00" * len(ct.tag)
    with pytest.raises(Exception):
        aes_gcm_decrypt(key, iv, ct.ciphertext, bad_tag, aad=b"context")
