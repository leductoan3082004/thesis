import dataclasses
import math
import secrets
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric import ec

from secure_aggregation.crypto import (
    aes_gcm_decrypt,
    aes_gcm_encrypt,
    derive_shared_key,
    generate_ecdh_keypair,
    generate_signing_keypair,
    prg_bytes,
    split_secret,
    combine_shares,
    verify_signature,
    sign_message,
)
from secure_aggregation.crypto.shamir import PRIME


SHARE_BYTES = 66  # enough to hold elements in the 521-bit field
DH_PRIV_BYTES = 32
B_SEED_BYTES = 32
MASK_BYTES_PER_COORD = 8


def _int_to_bytes(value: int, length: int) -> bytes:
    return int(value).to_bytes(length, byteorder="big")


def _bytes_to_int(data: bytes) -> int:
    return int.from_bytes(data, byteorder="big")


def _mask_from_seed(seed: bytes, length: int) -> List[int]:
    """Derive a vector of length elements using PRG output interpreted mod PRIME."""
    buf = prg_bytes(seed, length * MASK_BYTES_PER_COORD)
    coords: List[int] = []
    for i in range(length):
        segment = buf[i * MASK_BYTES_PER_COORD : (i + 1) * MASK_BYTES_PER_COORD]
        coords.append(int.from_bytes(segment, byteorder="big") % PRIME)
    return coords


def _vector_add(a: Sequence[int], b: Sequence[int]) -> List[int]:
    if len(a) != len(b):
        raise ValueError("Vector lengths must match")
    return [(x + y) % PRIME for x, y in zip(a, b)]


def _vector_sub(a: Sequence[int], b: Sequence[int]) -> List[int]:
    if len(a) != len(b):
        raise ValueError("Vector lengths must match")
    return [(x - y) % PRIME for x, y in zip(a, b)]


def _vector_zero(length: int) -> List[int]:
    return [0 for _ in range(length)]


@dataclass
class AdvertiseMessage:
    node_id: str
    c_public: bytes
    s_public: bytes
    signature: bytes
    signing_public: Optional[bytes] = None


@dataclass
class Round1Ciphertext:
    sender_id: str
    recipient_id: str
    iv: bytes
    ciphertext: bytes
    tag: bytes


@dataclass
class MaskedInput:
    node_id: str
    masked_vector: List[int]


@dataclass
class SurvivorSignature:
    node_id: str
    signature: bytes


@dataclass
class UnmaskingShares:
    node_id: str
    s_shares_for_dropouts: Dict[str, Tuple[int, int]]  # dropout_id -> (x, share)
    b_shares_for_survivors: Dict[str, Tuple[int, int]]  # survivor_id -> (x, share)


@dataclass
class SecureAggregationResult:
    survivors: List[str]
    aggregate_sum: List[int]
    aggregate_mean: List[float]


@dataclass
class SecureAggregationConfig:
    participants: List[str]
    threshold: int

    def __post_init__(self) -> None:
        if self.threshold <= 0 or self.threshold > len(self.participants):
            raise ValueError("Threshold must satisfy 0 < t <= |participants|")


class SecureAggregationNode:
    """
    Implements client-side steps for secure aggregation rounds 0-4.
    Designed for in-process orchestration; networking is left to callers.
    """

    def __init__(self, node_id: str, signing_private=None, signing_public=None) -> None:
        self.node_id = node_id
        if signing_private and signing_public:
            self.signing_private, self.signing_public = signing_private, signing_public
        else:
            keypair = generate_signing_keypair()
            self.signing_private, self.signing_public = keypair.private_key, keypair.public_key
        self.c_private = None
        self.c_public = b""
        self.s_private = None
        self.s_public = b""
        self._peer_adverts: Dict[str, AdvertiseMessage] = {}
        self._peer_shares: Dict[str, Tuple[int, int, int]] = {}  # peer_id -> (x, s_share, b_share)
        self._b_seed_bytes: bytes | None = None

    def advertise_keys(self) -> AdvertiseMessage:
        self.c_private, self.c_public = generate_ecdh_keypair()
        self.s_private, self.s_public = generate_ecdh_keypair()
        message = self.c_public + self.s_public
        signature = sign_message(self.signing_private, message)
        return AdvertiseMessage(
            node_id=self.node_id,
            c_public=self.c_public,
            s_public=self.s_public,
            signature=signature,
            signing_public=self.signing_public,
        )

    def receive_advertisements(self, adverts: Sequence[AdvertiseMessage]) -> None:
        for advert in adverts:
            self._peer_adverts[advert.node_id] = advert

    def _share_index(self, ordered_participants: Sequence[str]) -> int:
        if self.node_id not in ordered_participants:
            raise ValueError(f"Node {self.node_id} not part of participant list")
        return ordered_participants.index(self.node_id) + 1  # Shamir x coordinate

    def create_round1_ciphertexts(
        self, ordered_participants: Sequence[str], threshold: int
    ) -> List[Round1Ciphertext]:
        if not self.c_private or not self.s_private:
            raise ValueError("Call advertise_keys before Round 1")
        n = len(ordered_participants)
        s_secret_int = self.s_private.private_numbers().private_value
        b_seed_bytes = secrets.token_bytes(B_SEED_BYTES)
        b_seed_int = _bytes_to_int(b_seed_bytes)
        self._b_seed_bytes = b_seed_bytes
        s_shares = {x: y for x, y in split_secret(s_secret_int, n=n, t=threshold)}
        b_shares = {x: y for x, y in split_secret(b_seed_int, n=n, t=threshold)}
        ciphertexts: List[Round1Ciphertext] = []
        for participant in ordered_participants:
            advert = self._peer_adverts[participant]
            x_index = ordered_participants.index(participant) + 1
            s_share = s_shares[x_index]
            b_share = b_shares[x_index]
            payload = (
                x_index.to_bytes(2, "big")
                + _int_to_bytes(s_share, SHARE_BYTES)
                + _int_to_bytes(b_share, SHARE_BYTES)
            )
            key = derive_shared_key(self.c_private, advert.c_public, info=b"secure-agg/cipher")
            aad = f"{self.node_id}->{participant}".encode()
            ct = aes_gcm_encrypt(key, payload, aad=aad)
            ciphertexts.append(
                Round1Ciphertext(
                    sender_id=self.node_id,
                    recipient_id=participant,
                    iv=ct.iv,
                    ciphertext=ct.ciphertext,
                    tag=ct.tag,
                )
            )
        return ciphertexts

    def receive_round1_ciphertexts(self, payloads: Sequence[Round1Ciphertext]) -> None:
        for payload in payloads:
            advert = self._peer_adverts[payload.sender_id]
            key = derive_shared_key(self.c_private, advert.c_public, info=b"secure-agg/cipher")
            aad = f"{payload.sender_id}->{self.node_id}".encode()
            plaintext = aes_gcm_decrypt(key, payload.iv, payload.ciphertext, payload.tag, aad=aad)
            x = int.from_bytes(plaintext[:2], "big")
            s_share = _bytes_to_int(plaintext[2 : 2 + SHARE_BYTES])
            b_share = _bytes_to_int(plaintext[2 + SHARE_BYTES :])
            self._peer_shares[payload.sender_id] = (x, s_share, b_share)

    def _pairwise_mask(self, peer_id: str, vector_length: int) -> List[int]:
        peer_advert = self._peer_adverts[peer_id]
        seed = derive_shared_key(self.s_private, peer_advert.s_public, info=b"secure-agg/mask")
        mask = _mask_from_seed(seed, vector_length)
        if self.node_id > peer_id:
            return mask
        return [(-m) % PRIME for m in mask]

    def _self_mask(self, vector_length: int) -> List[int]:
        if self._b_seed_bytes is None:
            raise ValueError("Missing self-mask seed")
        return _mask_from_seed(self._b_seed_bytes, vector_length)

    def create_masked_input(self, model_vector: Sequence[int]) -> MaskedInput:
        vector = [v % PRIME for v in model_vector]
        masked = _vector_add(vector, self._self_mask(len(vector)))
        for peer in self._peer_shares:
            if peer == self.node_id:
                continue
            masked = _vector_add(masked, self._pairwise_mask(peer, len(vector)))
        return MaskedInput(node_id=self.node_id, masked_vector=masked)

    def sign_survivor_list(self, survivors: Sequence[str]) -> SurvivorSignature:
        message = ",".join(sorted(survivors)).encode()
        return SurvivorSignature(node_id=self.node_id, signature=sign_message(self.signing_private, message))

    def prepare_unmasking_payload(
        self, dropouts: Iterable[str], survivors: Iterable[str]
    ) -> UnmaskingShares:
        s_shares: Dict[str, Tuple[int, int]] = {}
        b_shares: Dict[str, Tuple[int, int]] = {}
        for dropout in dropouts:
            if dropout not in self._peer_shares:
                continue
            x, s_share, _ = self._peer_shares[dropout]
            s_shares[dropout] = (x, s_share)
        for survivor in survivors:
            if survivor not in self._peer_shares:
                # Self b-share is not held; skip and rely on other peers.
                continue
            x, _, b_share = self._peer_shares[survivor]
            b_shares[survivor] = (x, b_share)
        return UnmaskingShares(node_id=self.node_id, s_shares_for_dropouts=s_shares, b_shares_for_survivors=b_shares)


class SecureAggregationAggregator:
    """
    Central coordinator for the secure aggregation protocol. Assumes authenticated channels
    (long-term signing keys provided out-of-band by a TTP).
    """

    def __init__(self, config: SecureAggregationConfig, signing_public_keys: Optional[Mapping[str, bytes]] = None) -> None:
        if signing_public_keys is not None and set(config.participants) - set(signing_public_keys):
            missing = set(config.participants) - set(signing_public_keys)
            raise ValueError(f"Missing signing keys for {missing}")
        self.config = config
        self.signing_public_keys = dict(signing_public_keys or {})
        self.participants = list(config.participants)
        self.advertisements: Dict[str, AdvertiseMessage] = {}
        self.round1_mailboxes: MutableMapping[str, List[Round1Ciphertext]] = {p: [] for p in self.participants}
        self.masked_inputs: Dict[str, List[int]] = {}
        self.survivors: List[str] = []
        self.vector_length: int | None = None
        self._dropouts: List[str] = []

    def receive_advertisements(self, adverts: Sequence[AdvertiseMessage]) -> None:
        for advert in adverts:
            if advert.node_id not in self.participants:
                raise ValueError(f"Unexpected participant {advert.node_id}")
            if advert.node_id in self.advertisements:
                raise ValueError(f"Duplicate advertisement from {advert.node_id}")
            message = advert.c_public + advert.s_public
            expected_key = self.signing_public_keys.get(advert.node_id)
            derived_key = advert.signing_public or expected_key
            if expected_key and advert.signing_public and expected_key != advert.signing_public:
                raise ValueError(f"Signing key mismatch for {advert.node_id}")
            if not derived_key:
                raise ValueError(f"Missing signing key for {advert.node_id}")
            if not verify_signature(derived_key, message, advert.signature):
                raise ValueError(f"Invalid signature for {advert.node_id}")
            self.signing_public_keys[advert.node_id] = derived_key
            self.advertisements[advert.node_id] = advert
        if len(self.advertisements) < self.config.threshold:
            raise ValueError("Insufficient advertisements to meet threshold")

    def broadcast_advertisements(self) -> List[AdvertiseMessage]:
        return [self.advertisements[p] for p in self.participants if p in self.advertisements]

    def receive_round1_ciphertexts(self, payloads: Sequence[Round1Ciphertext]) -> None:
        for payload in payloads:
            if payload.recipient_id not in self.round1_mailboxes:
                raise ValueError(f"Unknown recipient {payload.recipient_id}")
            self.round1_mailboxes[payload.recipient_id].append(payload)

    def deliver_round1_ciphertexts(self, node_id: str) -> List[Round1Ciphertext]:
        return list(self.round1_mailboxes.get(node_id, []))

    def receive_masked_input(self, masked: MaskedInput) -> None:
        if masked.node_id in self.masked_inputs:
            raise ValueError(f"Duplicate masked input from {masked.node_id}")
        if self.vector_length is None:
            self.vector_length = len(masked.masked_vector)
        elif len(masked.masked_vector) != self.vector_length:
            raise ValueError("Masked input length mismatch")
        self.masked_inputs[masked.node_id] = [v % PRIME for v in masked.masked_vector]
        if len(self.masked_inputs) >= self.config.threshold:
            self.survivors = sorted(self.masked_inputs.keys())

    def broadcast_survivors(self) -> List[str]:
        if len(self.masked_inputs) < self.config.threshold:
            raise ValueError("Threshold not met for survivor broadcast")
        return list(self.survivors)

    def verify_survivor_signatures(self, signatures: Sequence[SurvivorSignature]) -> None:
        if not self.survivors:
            raise ValueError("No survivors recorded")
        expected_message = ",".join(sorted(self.survivors)).encode()
        signer_set = set()
        for sig in signatures:
            pub = self.signing_public_keys.get(sig.node_id)
            if not pub:
                raise ValueError(f"Unknown signer {sig.node_id}")
            if sig.node_id not in self.survivors:
                raise ValueError(f"Signature from non-survivor {sig.node_id}")
            if not verify_signature(pub, expected_message, sig.signature):
                raise ValueError(f"Invalid survivor signature from {sig.node_id}")
            signer_set.add(sig.node_id)
        if signer_set != set(self.survivors):
            raise ValueError("Missing survivor signatures")

    def receive_unmasking_shares(self, shares: Sequence[UnmaskingShares]) -> SecureAggregationResult:
        if not self.survivors:
            raise ValueError("Survivors must be established before unmasking")
        if len(shares) < self.config.threshold:
            raise ValueError("Not enough unmasking shares to reach threshold")
        dropouts = [p for p in self.participants if p not in self.survivors]
        self._dropouts = dropouts
        # Reconstruct s for dropouts
        dropout_s_priv: Dict[str, ec.EllipticCurvePrivateKey] = {}
        for dropout in dropouts:
            collected: List[Tuple[int, int]] = []
            for payload in shares:
                if dropout in payload.s_shares_for_dropouts:
                    collected.append(payload.s_shares_for_dropouts[dropout])
            if len(collected) < self.config.threshold:
                raise ValueError(f"Insufficient s-shares for dropout {dropout}")
            priv_bytes = combine_shares(collected, as_bytes_length=DH_PRIV_BYTES)
            priv_int = _bytes_to_int(priv_bytes)
            dropout_s_priv[dropout] = ec.derive_private_key(priv_int, ec.SECP256R1())
        # Reconstruct b for survivors
        survivor_b_seeds: Dict[str, bytes] = {}
        for survivor in self.survivors:
            collected: List[Tuple[int, int]] = []
            for payload in shares:
                if survivor in payload.b_shares_for_survivors:
                    collected.append(payload.b_shares_for_survivors[survivor])
            if len(collected) < self.config.threshold:
                raise ValueError(f"Insufficient b-shares for survivor {survivor}")
            seed_bytes = combine_shares(collected, as_bytes_length=B_SEED_BYTES)
            survivor_b_seeds[survivor] = seed_bytes
        # Aggregate masked inputs
        aggregate = _vector_zero(self.vector_length or 0)
        for survivor in self.survivors:
            aggregate = _vector_add(aggregate, self.masked_inputs[survivor])
        # Remove dropout pairwise masks
        for dropout in dropouts:
            dropout_s = dropout_s_priv[dropout]
            for survivor in self.survivors:
                seed = derive_shared_key(dropout_s, self.advertisements[survivor].s_public, info=b"secure-agg/mask")
                mask = _mask_from_seed(seed, self.vector_length)
                if dropout > survivor:
                    aggregate = _vector_add(aggregate, mask)
                else:
                    aggregate = _vector_sub(aggregate, mask)
        # Remove self masks for survivors
        for survivor in self.survivors:
            seed = survivor_b_seeds[survivor]
            mask = _mask_from_seed(seed, self.vector_length)
            aggregate = _vector_sub(aggregate, mask)
        mean = [val / len(self.survivors) for val in aggregate]
        return SecureAggregationResult(survivors=list(self.survivors), aggregate_sum=aggregate, aggregate_mean=mean)
