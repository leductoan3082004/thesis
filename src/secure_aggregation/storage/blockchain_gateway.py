"""
Blockchain gateway service for model anchoring with JWT authentication.

This service simulates a Hyperledger Fabric gateway by providing REST endpoints
for committing and retrieving data payloads, authenticated via Ed25519-signed JWTs.

Run with: uvicorn secure_aggregation.storage.blockchain_gateway:app --host 0.0.0.0 --port 9000
"""

import base64
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CommitRequest(BaseModel):
    """Request model for generic data commit."""

    payload: Dict[str, Any]


class CommitResponse(BaseModel):
    """Response model for data commit."""

    data_id: str
    submitted_at: str


class DataResponse(BaseModel):
    """Response model for data retrieval."""

    payload: Dict[str, Any]
    submitted_at: str


class ClusterModelRequest(BaseModel):
    """Request body for committing cluster-level models."""

    cluster_id: str
    payload: Dict[str, Any]
    round: Optional[int] = None


class ClusterModelResponse(BaseModel):
    """Response body for retrieving cluster-level models."""

    cluster_id: str
    round: int
    payload: Dict[str, Any]
    submitted_at: str


class ScopeModelRequest(BaseModel):
    """Request body for committing generic hierarchy models."""

    payload: Dict[str, Any]
    round: Optional[int] = None
    scope_round: Optional[int] = None
    scope_id: Optional[str] = None

    class Config:
        extra = "allow"


class ScopeModelResponse(BaseModel):
    """Response body for retrieving hierarchy models."""

    round: int
    payload: Dict[str, Any]
    submitted_at: str
    data_id: str
    scope_id: Optional[str] = None

    class Config:
        extra = "allow"


class DataStore:
    """Thread-safe storage for committed data with optional persistence."""

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        if self._storage_path:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            if self._storage_path.exists():
                self._load()
            else:
                self._save()

    def _load(self) -> None:
        try:
            self._data = json.loads(self._storage_path.read_text())
            logger.info(f"Loaded {len(self._data)} records from {self._storage_path}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to load data store: {e}")
            self._data = {}

    def _save(self) -> None:
        if not self._storage_path:
            return
        self._storage_path.write_text(json.dumps(self._data, indent=2))

    def commit(self, payload: Dict[str, Any]) -> tuple[str, str]:
        """Commit payload and return (data_id, submitted_at)."""
        data_id = str(uuid.uuid4())
        submitted_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._data[data_id] = {
                "payload": payload,
                "submitted_at": submitted_at,
            }
            self._save()

        logger.info(
            f"Committed data_id={data_id} cluster={payload.get('cluster_id')} round={payload.get('round')}"
        )
        return data_id, submitted_at

    def get(self, data_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(data_id)

    def list_all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"data_id": k, **v}
                for k, v in self._data.items()
            ]

    def find_scope_record(self, scope_key: str, scope_id: str, scope_round: int) -> Optional[Dict[str, Any]]:
        scope_type = f"{scope_key}_model"
        with self._lock:
            for data_id, record in self._data.items():
                payload = record.get("payload", {})
                if payload.get("type") != scope_type:
                    continue
                payload_scope_id = payload.get("scope_id") or payload.get(f"{scope_key}_id")
                if payload_scope_id != scope_id:
                    continue
                payload_round = payload.get("round")
                if payload_round is None:
                    payload_round = payload.get("scope_round")
                if payload_round != scope_round:
                    continue
                return {"data_id": data_id, **record}
        return None

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._save()


class JWTVerifier:
    """Verifies Ed25519-signed JWT tokens."""

    def __init__(self, public_keys_dir: str) -> None:
        self._public_keys: Dict[str, ed25519.Ed25519PublicKey] = {}
        self._keys_dir = Path(public_keys_dir)
        self._load_keys()

    def _load_keys(self) -> None:
        if not self._keys_dir.exists():
            logger.warning(f"Public keys directory {self._keys_dir} does not exist")
            return

        for key_file in self._keys_dir.glob("*_pk.pem"):
            try:
                key_bytes = key_file.read_bytes()
                public_key = serialization.load_pem_public_key(key_bytes)
                if isinstance(public_key, ed25519.Ed25519PublicKey):
                    identity = key_file.stem.replace("_pk", "")
                    self._public_keys[identity] = public_key
                    logger.info(f"Loaded public key for {identity}")
            except Exception as e:
                logger.error(f"Failed to load key from {key_file}: {e}")

        logger.info(f"Loaded {len(self._public_keys)} public keys")

    def reload_keys(self) -> None:
        self._public_keys.clear()
        self._load_keys()

    def verify(self, token: str) -> Dict[str, Any]:
        """Verify JWT and return claims. Raises HTTPException on failure."""
        parts = token.split(".")
        if len(parts) != 3:
            raise HTTPException(status_code=401, detail="Invalid JWT format")

        try:
            header_b64, payload_b64, sig_b64 = parts
            header = json.loads(self._b64_decode(header_b64))
            payload = json.loads(self._b64_decode(payload_b64))
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=401, detail=f"Invalid JWT encoding: {e}")

        if header.get("alg") != "EdDSA":
            raise HTTPException(status_code=401, detail="Unsupported algorithm")

        subject = payload.get("sub")
        if not subject or subject not in self._public_keys:
            raise HTTPException(status_code=401, detail=f"Unknown subject: {subject}")

        exp = payload.get("exp")
        if exp and time.time() > exp:
            raise HTTPException(status_code=401, detail="Token expired")

        unsigned = f"{header_b64}.{payload_b64}"
        signature = self._b64_decode(sig_b64)

        try:
            self._public_keys[subject].verify(signature, unsigned.encode("utf-8"))
        except InvalidSignature:
            raise HTTPException(status_code=401, detail="Invalid signature")

        return payload

    @staticmethod
    def _b64_decode(data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)


STORAGE_PATH = os.environ.get("GATEWAY_STORAGE_PATH", "/data/blockchain_gateway.json")
PUBLIC_KEYS_DIR = os.environ.get("GATEWAY_PUBLIC_KEYS_DIR", "/app/config/keys")

store = DataStore(storage_path=STORAGE_PATH)
verifier = JWTVerifier(public_keys_dir=PUBLIC_KEYS_DIR)

app = FastAPI(
    title="Blockchain Gateway",
    description="REST gateway for blockchain-based model anchoring with JWT authentication",
    version="1.0.0",
)


async def verify_jwt(authorization: str = Header(...)) -> Dict[str, Any]:
    """Dependency to verify JWT from Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[7:]
    return verifier.verify(token)


@app.post("/data/commit", response_model=CommitResponse)
async def commit_data(
    request: CommitRequest,
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> CommitResponse:
    """Commit data payload to the blockchain."""
    data_id, submitted_at = store.commit(request.payload)
    return CommitResponse(data_id=data_id, submitted_at=submitted_at)


def _get_typed_payload(data_id: str, expected_type: str) -> tuple[Dict[str, Any], str]:
    record = store.get(data_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Data not found")
    payload = record.get("payload", {})
    if payload.get("type") != expected_type:
        raise HTTPException(status_code=404, detail="Data not found")
    return payload, record["submitted_at"]


@app.post("/cluster/models", response_model=CommitResponse)
async def commit_cluster_model(
    request: ClusterModelRequest,
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> CommitResponse:
    """Commit cluster-level model metadata."""
    round_num = request.round if request.round is not None else request.payload.get("round")
    if round_num is None:
        raise HTTPException(status_code=422, detail="'round' field is required")
    payload_copy = dict(request.payload)
    payload_copy.pop("round", None)
    payload = {
        "type": "cluster_model",
        "cluster_id": request.cluster_id,
        "round": int(round_num),
        "payload": payload_copy,
    }
    data_id, submitted_at = store.commit(payload)
    return CommitResponse(data_id=data_id, submitted_at=submitted_at)


@app.get("/cluster/models/{data_id}", response_model=ClusterModelResponse)
async def get_cluster_model(
    data_id: str,
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> ClusterModelResponse:
    """Retrieve cluster-level model metadata."""
    payload, submitted_at = _get_typed_payload(data_id, "cluster_model")
    round_val = payload.get("round")
    if round_val is None:
        round_val = payload.get("payload", {}).get("round")
    if round_val is None:
        raise HTTPException(status_code=500, detail="Cluster record missing 'round'")
    return ClusterModelResponse(
        cluster_id=payload["cluster_id"],
        round=int(round_val),
        payload=payload["payload"],
        submitted_at=submitted_at,
    )


@app.post("/{scope_name}/models", response_model=CommitResponse)
async def commit_scope_model(
    request: ScopeModelRequest,
    scope_name: str = Path(..., regex="^(?!cluster$).+"),
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> CommitResponse:
    """Commit hierarchy-level model metadata."""
    scope_key = scope_name.lower()
    if scope_key == "cluster":
        raise HTTPException(status_code=400, detail="Use /cluster/models for cluster commits")
    round_num = request.round if request.round is not None else request.scope_round
    if round_num is None:
        raise HTTPException(status_code=422, detail="'round' field is required")
    identifier_field = f"{scope_key}_id"
    scope_identifier = getattr(request, identifier_field, None) or request.scope_id
    if not scope_identifier:
        raise HTTPException(status_code=422, detail=f"'{identifier_field}' field is required")
    payload_copy = dict(request.payload)
    payload_copy.pop("round", None)
    payload_copy.pop("scope_round", None)
    payload = {
        "type": f"{scope_key}_model",
        "scope": scope_key,
        "scope_id": scope_identifier,
        identifier_field: scope_identifier,
        "round": int(round_num),
        "payload": payload_copy,
    }
    data_id, submitted_at = store.commit(payload)
    return CommitResponse(data_id=data_id, submitted_at=submitted_at)


@app.get("/{scope_name}/models/{data_id}", response_model=ScopeModelResponse)
async def get_scope_model_by_id(
    data_id: str,
    scope_name: str = Path(..., regex="^(?!cluster$).+"),
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> ScopeModelResponse:
    """Retrieve hierarchy-level model metadata by data_id."""
    scope_key = scope_name.lower()
    payload, submitted_at = _get_typed_payload(data_id, f"{scope_key}_model")
    identifier_field = f"{scope_key}_id"
    identifier_value = payload.get(identifier_field) or payload.get("scope_id")
    round_val = payload.get("round")
    if round_val is None:
        round_val = payload.get("scope_round")
    if round_val is None:
        raise HTTPException(status_code=500, detail="Scope record missing 'round'")
    response_kwargs: Dict[str, Any] = {
        "round": int(round_val),
        "payload": payload["payload"],
        "submitted_at": submitted_at,
        "data_id": data_id,
        "scope_id": payload.get("scope_id"),
    }
    if identifier_value and identifier_field != "scope_id":
        response_kwargs[identifier_field] = identifier_value
    return ScopeModelResponse(
        **response_kwargs,
    )


@app.get("/{scope_name}/models", response_model=ScopeModelResponse)
async def get_scope_model(
    scope_name: str = Path(..., regex="^(?!cluster$).+"),
    round_num: int = Query(..., alias="round"),
    request: Request,
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> ScopeModelResponse:
    """Retrieve hierarchy-level model metadata by scope identifier and round."""
    scope_key = scope_name.lower()
    identifier_field = f"{scope_key}_id"
    query_params = request.query_params if request is not None else {}
    scope_identifier = query_params.get(identifier_field) or query_params.get("scope_id")
    if not scope_identifier:
        raise HTTPException(status_code=422, detail=f"Query parameter '{identifier_field}' is required")
    record = store.find_scope_record(scope_key, scope_identifier, round_num)
    if record is None:
        raise HTTPException(status_code=404, detail="Model not found")
    payload = record["payload"]
    response_kwargs: Dict[str, Any] = {
        "round": payload.get("round") or payload.get("scope_round"),
        "payload": payload["payload"],
        "submitted_at": record["submitted_at"],
        "data_id": record["data_id"],
        "scope_id": payload.get("scope_id"),
    }
    identifier_value = payload.get(identifier_field) or payload.get("scope_id")
    if response_kwargs["round"] is None:
        raise HTTPException(status_code=500, detail="Stored scope model missing 'round'")
    response_kwargs["round"] = int(response_kwargs["round"])
    if identifier_value and identifier_field != "scope_id":
        response_kwargs[identifier_field] = identifier_value
    return ScopeModelResponse(**response_kwargs)


@app.get("/data/{data_id}", response_model=DataResponse)
async def get_data(
    data_id: str,
    claims: Dict[str, Any] = Depends(verify_jwt),
) -> DataResponse:
    """Retrieve data payload by ID."""
    record = store.get(data_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Data not found")
    return DataResponse(payload=record["payload"], submitted_at=record["submitted_at"])


@app.get("/data")
async def list_data(claims: Dict[str, Any] = Depends(verify_jwt)) -> List[Dict[str, Any]]:
    """List all committed data."""
    return store.list_all()


@app.delete("/data", status_code=204)
async def clear_data(claims: Dict[str, Any] = Depends(verify_jwt)) -> None:
    """Clear all data (for testing)."""
    store.clear()


@app.post("/keys/reload", status_code=200)
async def reload_keys() -> Dict[str, str]:
    """Reload public keys from disk."""
    verifier.reload_keys()
    return {"status": "reloaded"}


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}
