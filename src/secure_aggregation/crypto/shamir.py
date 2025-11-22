import secrets
from typing import Iterable, List, Sequence, Tuple

# Large prime field; ample headroom for keys/seeds.
PRIME = 2**521 - 1


def _to_int(secret: bytes | int) -> int:
    if isinstance(secret, int):
        value = secret
    else:
        value = int.from_bytes(secret, byteorder="big")
    if value >= PRIME:
        raise ValueError("Secret is too large for configured prime field")
    return value


def _eval_polynomial(coeffs: Sequence[int], x: int) -> int:
    acc = 0
    for power, coeff in enumerate(coeffs):
        acc = (acc + coeff * pow(x, power, PRIME)) % PRIME
    return acc


def split_secret(secret: bytes | int, n: int, t: int) -> List[Tuple[int, int]]:
    """
    Split a secret into n shares with threshold t using Shamir's Secret Sharing.
    """
    if not (2 <= t <= n):
        raise ValueError("Threshold t must satisfy 2 <= t <= n")
    secret_int = _to_int(secret)
    coeffs = [secret_int] + [secrets.randbelow(PRIME) for _ in range(t - 1)]
    shares: List[Tuple[int, int]] = []
    for x in range(1, n + 1):
        shares.append((x, _eval_polynomial(coeffs, x)))
    return shares


def combine_shares(shares: Iterable[Tuple[int, int]], as_bytes_length: int | None = None) -> bytes | int:
    """
    Reconstruct the secret from shares using Lagrange interpolation at x=0.
    """
    share_list = list(shares)
    if len(share_list) == 0:
        raise ValueError("At least one share is required to reconstruct")
    x_s, y_s = zip(*share_list)
    if len(set(x_s)) != len(x_s):
        raise ValueError("Duplicate share indices detected")
    secret = 0
    for j, (xj, yj) in enumerate(share_list):
        numerator = 1
        denominator = 1
        for m, (xm, _) in enumerate(share_list):
            if m == j:
                continue
            numerator = (numerator * (-xm)) % PRIME
            denominator = (denominator * (xj - xm)) % PRIME
        inv = pow(denominator, -1, PRIME)
        secret = (secret + yj * numerator * inv) % PRIME
    if as_bytes_length is not None:
        return int(secret).to_bytes(as_bytes_length, byteorder="big")
    return int(secret)
