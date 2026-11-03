"""Exception hierarchy.

Every error raised by Rafidain derives from :class:`RafidainError`, carrying a
stable machine-readable ``code`` and an HTTP status so the API layer can map
domain failures to responses without ``if/elif`` on exception types.

This is the ``E`` in the error-handling strategy used across layers:
domain/application raise semantic errors; only ``api`` knows about HTTP.
"""

from __future__ import annotations


class RafidainError(Exception):
    """Base class for all domain-level failures."""

    code = "rafidain_error"
    http_status = 500

    def __init__(self, message: str = "", *, detail: dict | None = None) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "detail": self.detail}}


# --- configuration & environment ---------------------------------------------


class ConfigError(RafidainError):
    code = "config_error"
    http_status = 500


class DependencyMissingError(ConfigError):
    """A optional dependency (torch, seed-vc repo, webrtcvad...) is required."""

    code = "dependency_missing"
    http_status = 503


# --- engine -------------------------------------------------------------------


class EngineError(RafidainError):
    code = "engine_error"
    http_status = 500


class EngineNotReadyError(EngineError):
    code = "engine_not_ready"
    http_status = 503


class EngineBusyError(EngineError):
    """The engine concurrency budget is exhausted."""

    code = "engine_busy"
    http_status = 503


class ReferenceAudioError(EngineError):
    """The reference/voice audio cannot be used to condition the model."""

    code = "invalid_reference_audio"
    http_status = 400


# --- sessions -----------------------------------------------------------------


class SessionError(RafidainError):
    code = "session_error"
    http_status = 400


class SessionNotFoundError(SessionError):
    code = "session_not_found"
    http_status = 404


class SessionExpiredError(SessionError):
    code = "session_expired"
    http_status = 410


class SessionLimitError(SessionError):
    code = "session_limit_reached"
    http_status = 429


class InvalidSessionStateError(SessionError):
    code = "invalid_session_state"
    http_status = 409


# --- voices -------------------------------------------------------------------


class VoiceError(RafidainError):
    code = "voice_error"
    http_status = 400


class VoiceNotFoundError(VoiceError):
    code = "voice_not_found"
    http_status = 404


class VoiceValidationError(VoiceError):
    code = "voice_validation_failed"
    http_status = 422


# --- jobs (fine-tuning) --------------------------------------------------------


class JobError(RafidainError):
    code = "job_error"
    http_status = 400


class JobNotFoundError(JobError):
    code = "job_not_found"
    http_status = 404


# --- API / transport ------------------------------------------------------------


class AuthenticationError(RafidainError):
    code = "unauthorized"
    http_status = 401


class ProtocolError(RafidainError):
    """Malformed WebSocket control message or binary payload."""

    code = "protocol_error"
    http_status = 400
