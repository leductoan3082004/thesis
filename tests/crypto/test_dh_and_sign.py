from secure_aggregation.crypto import (
    derive_shared_key,
    generate_ecdh_keypair,
    generate_signing_keypair,
    sign_message,
    verify_signature,
)


def test_ecdh_shared_key_agreement() -> None:
    priv_a, pub_a = generate_ecdh_keypair()
    priv_b, pub_b = generate_ecdh_keypair()
    shared_ab = derive_shared_key(priv_a, pub_b)
    shared_ba = derive_shared_key(priv_b, pub_a)
    assert shared_ab == shared_ba
    assert len(shared_ab) == 32


def test_ed25519_sign_and_verify() -> None:
    priv, pub = generate_signing_keypair()
    message = b"round-0-list"
    sig = sign_message(priv, message)
    assert verify_signature(pub, message, sig)
    assert not verify_signature(pub, message + b"tamper", sig)
