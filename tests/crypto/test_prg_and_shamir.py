import pytest

from secure_aggregation.crypto import combine_shares, prg_bytes, split_secret


def test_prg_is_deterministic_and_length() -> None:
    seed = b"seed-123"
    out1 = prg_bytes(seed, 64)
    out2 = prg_bytes(seed, 64)
    assert out1 == out2
    assert len(out1) == 64
    assert prg_bytes(seed + b"x", 64) != out1


def test_shamir_split_and_reconstruct_bytes() -> None:
    secret = b"mask-seed"
    shares = split_secret(secret, n=5, t=3)
    recovered = combine_shares(shares[:3], as_bytes_length=len(secret))
    assert recovered == secret


def test_shamir_detects_duplicate_indices() -> None:
    with pytest.raises(ValueError):
        combine_shares([(1, 2), (1, 3)])
