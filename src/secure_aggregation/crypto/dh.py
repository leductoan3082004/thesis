from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


@dataclass
class DHKeyPair:
    """ECDH keypair with bytes representation."""
    private_key: bytes
    public_key: bytes


def generate_dh_keypair() -> DHKeyPair:
    """Generate an ECDH P-256 keypair. Returns DHKeyPair with raw bytes."""
    private_key_obj = ec.generate_private_key(ec.SECP256R1())
    private_bytes = private_key_obj.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_bytes = private_key_obj.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return DHKeyPair(private_key=private_bytes, public_key=public_bytes)


def generate_ecdh_keypair() -> tuple[ec.EllipticCurvePrivateKey, bytes]:
    """
    Generate an ECDH P-256 keypair. Returns (private_key, public_bytes).
    Public bytes are uncompressed X9.62 points.
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    return private_key, public_bytes


def load_public_key_bytes(data: bytes) -> ec.EllipticCurvePublicKey:
    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), data)


def derive_shared_key(
    private_key: ec.EllipticCurvePrivateKey, peer_public_bytes: bytes, length: int = 32, info: bytes = b"secure-agg/dh"
) -> bytes:
    """
    Derive a symmetric key via ECDH + HKDF-SHA256.
    """
    peer_pub = load_public_key_bytes(peer_public_bytes)
    shared = private_key.exchange(ec.ECDH(), peer_pub)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info)
    return hkdf.derive(shared)


def agree(private_key_bytes: bytes, peer_public_bytes: bytes) -> bytes:
    """
    Perform ECDH key agreement using PEM-encoded private key bytes and peer's public key bytes.
    Returns the derived shared secret.
    """
    # Load private key from PEM bytes
    private_key_obj = serialization.load_pem_private_key(private_key_bytes, password=None)
    if not isinstance(private_key_obj, ec.EllipticCurvePrivateKey):
        raise ValueError("Invalid private key type")

    # Derive shared key
    return derive_shared_key(private_key_obj, peer_public_bytes)
