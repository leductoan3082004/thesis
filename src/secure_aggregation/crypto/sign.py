from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


@dataclass
class SigningKeyPair:
    """Ed25519 signing keypair."""
    private_key: bytes
    public_key: bytes


def generate_signing_keypair() -> SigningKeyPair:
    """
    Generate an Ed25519 signing keypair.
    Returns SigningKeyPair with both keys as raw bytes.
    """
    private_key_obj = Ed25519PrivateKey.generate()
    private_bytes = private_key_obj.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_bytes = private_key_obj.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw
    )
    return SigningKeyPair(private_key=private_bytes, public_key=public_bytes)


def sign_message(private_key: bytes, message: bytes) -> bytes:
    """Sign a message using Ed25519 private key bytes."""
    private_key_obj = Ed25519PrivateKey.from_private_bytes(private_key)
    return private_key_obj.sign(message)


def verify_signature(public_key_bytes: bytes, message: bytes, signature: bytes) -> bool:
    try:
        pub = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        pub.verify(signature, message)
        return True
    except InvalidSignature:
        return False
