"""Bounded, in-memory accounting for the post-OutputIO result seal tail."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

_CHUNK_BYTES = 1_048_576


class SealLedgerError(RuntimeError):
    """A seal action was rejected or the ledger became permanently poisoned."""


class SealLedger:
    """Run-directory-only hash, line-count and exclusive JSON-write actions.

    This ledger is deliberately in-memory: the caller's independent fatal
    writer persists its snapshot after a failure, avoiding a self-sealing log.
    """

    def __init__(self, directory, max_actions):
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        requested = Path(directory).absolute()
        if requested.is_symlink() or self._is_junction(requested):
            raise PermissionError("seal directory cannot be a link or junction")
        self.directory = requested.resolve(strict=True)
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)
        self.max_actions = max_actions
        self.actions = []
        self.poisoned = False
        self._identities = set()
        self._total_read_bytes = 0
        self._total_write_bytes = 0
        self._unknown_operations = 0
        self._first_error = None

    @staticmethod
    def _is_junction(path):
        return bool(hasattr(path, "is_junction") and path.is_junction())

    @staticmethod
    def _display_name(name):
        return name if isinstance(name, str) else f"<{type(name).__name__}>"

    def _reserve(self, name, operation):
        if self.poisoned:
            error = SealLedgerError("seal ledger is permanently poisoned")
            error.input_consistency_seal_partial_record = self.snapshot()
            raise error
        if len(self.actions) >= self.max_actions:
            self.poisoned = True
            error = SealLedgerError("seal action cap exceeded")
            self._first_error = {
                "stage": "reserve",
                "type": type(error).__name__,
                "message": str(error),
                "name": self._display_name(name),
                "operation": operation,
            }
            error.input_consistency_seal_partial_record = self.snapshot()
            raise error
        action = {
            "action_id": len(self.actions),
            "name": self._display_name(name),
            "operation": operation,
            "status": "attempted",
            "stage": "reserve",
            "read_bytes_returned": 0,
            "write_bytes_returned": 0,
            "read_operations_attempted": 0,
            "read_operations_returned": 0,
            "write_operations_attempted": 0,
            "write_operations_returned": 0,
            "unknown_operations": 0,
            "short_writes": 0,
            "hash_prefix_sha256": hashlib.sha256().hexdigest()
            if operation in {"hash_file", "count_lines"}
            else None,
            "durability_unknown": False,
            "cleanup_errors": [],
        }
        self.actions.append(action)
        identity = (self._display_name(name), operation)
        if identity in self._identities:
            error = SealLedgerError("duplicate seal action identity; retries are forbidden")
            self._fail(action, error, stage="identity")
            raise error
        self._identities.add(identity)
        return action

    def _safe_path(self, name):
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or ":" in name
            or name.startswith("/")
            or "//" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))
        ):
            raise PermissionError("seal target must be a normalized relative path")
        pure = PurePosixPath(name)
        if pure.is_absolute() or pure.as_posix() != name:
            raise PermissionError("seal target must be a normalized relative path")
        target = self.directory.joinpath(*pure.parts)
        current = self.directory
        for part in pure.parts:
            current = current / part
            if current.is_symlink() or self._is_junction(current):
                raise PermissionError("seal target cannot traverse a link or junction")
        try:
            target.resolve(strict=False).relative_to(self.directory)
        except (OSError, ValueError) as error:
            raise PermissionError("seal target escapes its run directory") from error
        return target

    def _fail(self, action, error, *, stage, unknown=0, durability_unknown=False):
        action.update(
            status="failed",
            stage=stage,
            error_type=type(error).__name__,
            error_message=str(error),
            durability_unknown=bool(durability_unknown),
        )
        if unknown:
            action["unknown_operations"] += unknown
            self._unknown_operations += unknown
        self.poisoned = True
        if self._first_error is None:
            self._first_error = {
                "action_id": action["action_id"],
                "name": action["name"],
                "operation": action["operation"],
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
            }
        error.input_consistency_seal_partial_record = self.snapshot()

    def _perform_read(self, name, operation, callback):
        action = self._reserve(name, operation)
        digest = hashlib.sha256()
        stream = None
        primary = None
        stage = "path_validation"
        try:
            target = self._safe_path(name)
            stage = "stat_before"
            size_before = target.stat().st_size
            action["size_bytes_before"] = size_before
            stage = "open"
            stream = target.open("rb")
            stage = "read"
            value = callback(stream, action, digest)
            stage = "size_validation"
            size_after = target.stat().st_size
            action["size_bytes_after"] = size_after
            if action["read_bytes_returned"] != size_before or size_after != size_before:
                raise OSError("seal read byte count or file size changed during action")
            stage = "close"
            closing_stream, stream = stream, None
            closing_stream.close()
            stream = None
            action["status"] = "returned"
            action["stage"] = "complete"
            action["hash_prefix_sha256"] = digest.hexdigest()
            return value
        except BaseException as error:  # noqa: BLE001 - every interrupted read poisons the seal action
            primary = error
            if stage == "read":
                self._fail(action, error, stage=stage, unknown=1)
            else:
                self._fail(action, error, stage=stage, durability_unknown=stage == "close")
        finally:
            if stream is not None:
                try:
                    stream.close()
                except BaseException as close_error:  # noqa: BLE001 - close failure is part of the seal evidence
                    action["cleanup_errors"].append(
                        {"type": type(close_error).__name__, "message": str(close_error)}
                    )
                    action["durability_unknown"] = True
                    if primary is None:
                        primary = close_error
                        self._fail(action, close_error, stage="close", durability_unknown=True)
        if primary is not None:
            self._attach_partial(primary)
            raise primary

    def _record_read(self, action, digest, block):
        if not isinstance(block, bytes):
            raise TypeError("seal read must return bytes")
        action["read_operations_returned"] += 1
        action["read_bytes_returned"] += len(block)
        self._total_read_bytes += len(block)
        digest.update(block)
        action["hash_prefix_sha256"] = digest.hexdigest()
        return block

    def hash_file(self, name):
        def read_all(stream, action, digest):
            while True:
                action["read_operations_attempted"] += 1
                block = stream.read(_CHUNK_BYTES)
                self._record_read(action, digest, block)
                if not block:
                    break
            return digest.hexdigest()

        return self._perform_read(name, "hash_file", read_all)

    def count_lines(self, name):
        def count(stream, action, digest):
            lines, last_byte = 0, None
            while True:
                action["read_operations_attempted"] += 1
                block = stream.read(_CHUNK_BYTES)
                self._record_read(action, digest, block)
                if not block:
                    break
                lines += block.count(b"\n")
                last_byte = block[-1]
            if last_byte is not None and last_byte != 0x0A:
                lines += 1
            action["ends_with_newline"] = last_byte == 0x0A
            return lines

        return self._perform_read(name, "count_lines", count)

    def _attach_partial(self, error):
        error.input_consistency_seal_partial_record = self.snapshot()

    def write_json(self, name, value):
        action = self._reserve(name, "write_json")
        stream = None
        primary = None
        stage = "path_validation"
        try:
            target = self._safe_path(name)
            stage = "encode"
            payload = (
                json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            ).encode("utf-8")
            stage = "open"
            stream = target.open("xb")
            stage = "write"
            action["write_operations_attempted"] += 1
            try:
                written = stream.write(payload)
            except BaseException:
                action["unknown_operations"] += 1
                self._unknown_operations += 1
                raise
            if type(written) is not int or written < 0 or written > len(payload):
                action["unknown_operations"] += 1
                self._unknown_operations += 1
                raise OSError("seal write returned an invalid byte count")
            action["write_operations_returned"] += 1
            action["write_bytes_returned"] += written
            self._total_write_bytes += written
            if written != len(payload):
                action["short_writes"] += 1
                raise OSError("short write while saving sealed JSON")
            stage = "flush"
            stream.flush()
            stage = "fsync"
            os.fsync(stream.fileno())
            stage = "close"
            closing_stream, stream = stream, None
            closing_stream.close()
            action["status"] = "returned"
            action["stage"] = "complete"
            return written
        except BaseException as error:  # noqa: BLE001 - every interrupted write or durability stage poisons
            primary = error
            self._fail(
                action, error, stage=stage, durability_unknown=stage in {"flush", "fsync", "close"}
            )
        finally:
            if stream is not None:
                try:
                    stream.close()
                except BaseException as close_error:  # noqa: BLE001 - retain both primary and close failures
                    action["cleanup_errors"].append(
                        {"type": type(close_error).__name__, "message": str(close_error)}
                    )
                    action["durability_unknown"] = True
                    if primary is None:
                        primary = close_error
                        self._fail(action, close_error, stage="close", durability_unknown=True)
        if primary is not None:
            self._attach_partial(primary)
            raise primary

    def snapshot(self):
        completed = sum(action["status"] in {"returned", "failed"} for action in self.actions)
        pending = sum(action["status"] == "attempted" for action in self.actions)
        return copy.deepcopy(
            {
                "directory": self.directory.as_posix(),
                "max_actions": self.max_actions,
                "action_count": len(self.actions),
                "transition_records": 2 * completed + pending,
                "actual_read_bytes": self._total_read_bytes,
                "actual_write_bytes": self._total_write_bytes,
                "unknown_operations": self._unknown_operations,
                "poisoned": self.poisoned,
                "first_error": self._first_error,
                "actions": self.actions,
            }
        )
