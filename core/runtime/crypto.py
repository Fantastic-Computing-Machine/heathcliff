"""AES-GCM encryption for credentials kept outside the event journal."""

from __future__ import annotations

import base64
import os


class CredentialCipher:
    def __init__(self, key: str | None = None) -> None:
        raw = key or os.getenv("RUNTIME_STATE_ENCRYPTION_KEY")
        if not raw:
            raise ValueError("RUNTIME_STATE_ENCRYPTION_KEY is required")
        decoded = base64.urlsafe_b64decode(raw)
        if len(decoded) != 32:
            raise ValueError(
                "RUNTIME_STATE_ENCRYPTION_KEY must encode exactly 32 bytes"
            )
        self._key = decoded

    def encrypt(self, plaintext: bytes, aad: bytes = b"heathcliff") -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = os.urandom(12)
        return nonce + AESGCM(self._key).encrypt(nonce, plaintext, aad)

    def decrypt(self, ciphertext: bytes, aad: bytes = b"heathcliff") -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        return AESGCM(self._key).decrypt(ciphertext[:12], ciphertext[12:], aad)
