#!/usr/bin/env python3
"""Generate Ed25519 key pairs for blockchain gateway authentication."""

import argparse
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519


def generate_keypair(identity: str, output_dir: Path) -> None:
    """Generate Ed25519 key pair for a given identity."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    sk_path = output_dir / f"{identity}_sk.pem"
    pk_path = output_dir / f"{identity}_pk.pem"

    sk_path.write_bytes(private_pem)
    pk_path.write_bytes(public_pem)

    print(f"Generated keys for {identity}:")
    print(f"  Private key: {sk_path}")
    print(f"  Public key:  {pk_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ed25519 key pairs for trainers")
    parser.add_argument(
        "--num-trainers",
        type=int,
        default=6,
        help="Number of trainer key pairs to generate (default: 6)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("config/keys"),
        help="Output directory for keys (default: config/keys)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="trainer-node",
        help="Prefix for trainer identity names (default: trainer-node)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate keys starting from index 0 to match node configs.
    for i in range(args.num_trainers):
        identity = f"{args.prefix}-{i:03d}"
        generate_keypair(identity, args.output_dir)

    print(f"\nGenerated {args.num_trainers} key pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
