"""Measure Path.open activity confined to one A17 result directory.

The manager is an engineering boundary, not a source reader: paths outside the
designated result tree pass through unchanged. Its own JSONL journal bypasses
the patch and is excluded from measurement.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from pathlib import Path


class OutputIOError(RuntimeError):
    """A measured I/O failure poisoned this run's output boundary."""

    def __init__(self, message, *, partial_record=None, cause=None):
        super().__init__(message)
        self.partial_record = partial_record or {}
        if cause is not None:
            self.__cause__ = cause


class _MeasuredStream:
    """Delegate a file handle while measuring delivered read/write bytes."""

    def __init__(self, manager, stream, record):
        self._manager = manager
        self._stream = stream
        self._record = record
        self._closed = False

    def __getattr__(self, name):
        if name == "buffer":
            return _MeasuredBuffer(self, self._stream.buffer)
        if name == "raw":
            raise AttributeError(f"{name} is hidden by the output I/O meter")
        return getattr(self._stream, name)

    @property
    def closed(self):
        return self._stream.closed

    def _fail_operation(self, operation, error, *, unknown=False):
        self._manager._operation_failed(self._record, operation, error, unknown=unknown)

    def _encode_text(self, value):
        encoding = getattr(self._stream, "encoding", None) or "utf-8"
        errors = getattr(self._stream, "errors", None) or "strict"
        newline = self._record["newline"]
        if newline is None and os.linesep != "\n":
            value = value.replace("\n", os.linesep)
        elif newline not in (None, "", "\n"):
            value = value.replace("\n", newline)
        return len(value.encode(encoding, errors=errors))

    @staticmethod
    def _returned_bytes(value):
        if isinstance(value, str):
            return None
        try:
            return memoryview(value).nbytes
        except TypeError:
            return 0

    def read(self, size=-1):
        self._manager._check_open(self._record)
        self._record["read_calls_attempted"] += 1
        if isinstance(size, int) and size >= 0:
            self._record["read_requested_units"] += size
        else:
            self._record["read_unbounded_calls"] += 1
        try:
            value = self._stream.read(size)
            byte_count = (
                self._encode_text(value) if isinstance(value, str) else self._returned_bytes(value)
            )
            if byte_count is None:
                raise TypeError("text read did not return a string")
            self._record["read_calls_returned"] += 1
            self._record["read_bytes_returned"] += byte_count
            if isinstance(size, int) and size >= 0 and len(value) < size:
                self._record["short_reads"] += 1
            self._manager._totals["read_calls_returned"] += 1
            self._manager._totals["read_bytes_returned"] += byte_count
            return value
        except BaseException as error:
            self._record["read_bytes_unknown_operations"] += 1
            self._manager._totals["read_bytes_unknown_operations"] += 1
            self._fail_operation("read", error, unknown=True)
            raise

    def read1(self, size=-1):
        method = getattr(self._stream, "read1", None)
        if method is None:
            return self.read(size)
        return self._read_method("read1", method, size)

    def readinto(self, buffer):
        method = getattr(self._stream, "readinto", None)
        if method is None:
            raise AttributeError("readinto")
        return self._readinto_method("readinto", method, buffer)

    def readinto1(self, buffer):
        method = getattr(self._stream, "readinto1", None)
        if method is None:
            return self.readinto(buffer)
        return self._readinto_method("readinto1", method, buffer)

    def _read_method(self, name, method, size):
        self._manager._check_open(self._record)
        self._record["read_calls_attempted"] += 1
        if isinstance(size, int) and size >= 0:
            self._record["read_requested_units"] += size
        else:
            self._record["read_unbounded_calls"] += 1
        try:
            value = method(size)
            count = (
                self._encode_text(value) if isinstance(value, str) else self._returned_bytes(value)
            )
            if count is None:
                raise TypeError(f"{name} did not return bytes or text")
            self._record["read_calls_returned"] += 1
            self._record["read_bytes_returned"] += count
            if isinstance(size, int) and size >= 0 and len(value) < size:
                self._record["short_reads"] += 1
            self._manager._totals["read_calls_returned"] += 1
            self._manager._totals["read_bytes_returned"] += count
            return value
        except BaseException as error:
            self._record["read_bytes_unknown_operations"] += 1
            self._manager._totals["read_bytes_unknown_operations"] += 1
            self._fail_operation(name, error, unknown=True)
            raise

    def _readinto_method(self, name, method, buffer):
        self._manager._check_open(self._record)
        size = memoryview(buffer).nbytes
        self._record["read_calls_attempted"] += 1
        self._record["read_requested_units"] += size
        try:
            count = method(buffer)
            if type(count) is not int or not 0 <= count <= size:
                raise OSError("underlying readinto returned an invalid byte count")
            self._record["read_calls_returned"] += 1
            self._record["read_bytes_returned"] += count
            self._manager._totals["read_calls_returned"] += 1
            self._manager._totals["read_bytes_returned"] += count
            if count < size:
                self._record["short_reads"] += 1
            return count
        except BaseException as error:
            self._record["read_bytes_unknown_operations"] += 1
            self._manager._totals["read_bytes_unknown_operations"] += 1
            self._fail_operation(name, error, unknown=True)
            raise

    def readline(self, size=-1):
        return self._read_method("readline", self._stream.readline, size)

    def readlines(self, hint=-1):
        lines = []
        consumed = 0
        while hint < 0 or consumed < hint:
            line = self.readline()
            if not line:
                break
            lines.append(line)
            consumed += len(line)
        return lines

    def __iter__(self):
        self._manager._check_open(self._record)
        return self

    def __next__(self):
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def write(self, value):
        self._manager._check_open(self._record)
        if isinstance(value, str):
            requested = self._encode_text(value)
            text = True
        else:
            try:
                requested = memoryview(value).nbytes
            except TypeError as error:
                raise TypeError("write expects text or a bytes-like object") from error
            text = False
        self._record["write_calls_attempted"] += 1
        self._record["write_bytes_requested"] += requested
        known_return = False
        try:
            result = self._stream.write(value)
            if type(result) is not int or result < 0:
                raise OSError("underlying write returned no valid count")
            if text:
                actual = self._encode_text(value[:result])
            else:
                actual = result
            self._record["write_calls_returned"] += 1
            self._record["write_bytes_returned"] += actual
            self._manager._totals["write_calls_returned"] += 1
            self._manager._totals["write_bytes_returned"] += actual
            known_return = True
            if actual != requested:
                self._record["short_writes"] += 1
                raise OSError("underlying output write returned a short or invalid count")
            return result
        except BaseException as error:
            if not known_return:
                self._record["write_bytes_unknown_operations"] += 1
                self._manager._totals["write_bytes_unknown_operations"] += 1
            self._fail_operation("write", error, unknown=not known_return)
            raise

    def writelines(self, lines):
        self._manager._check_open(self._record)
        requested = 0

        def measured_lines():
            nonlocal requested
            for item in lines:
                requested += (
                    self._encode_text(item) if isinstance(item, str) else memoryview(item).nbytes
                )
                yield item

        self._record["write_calls_attempted"] += 1
        try:
            self._stream.writelines(measured_lines())
            self._record["write_bytes_requested"] += requested
            self._record["write_bytes_returned"] += requested
            self._record["write_calls_returned"] += 1
            self._manager._totals["write_bytes_returned"] += requested
            self._manager._totals["write_calls_returned"] += 1
            return
        except BaseException as error:
            self._record["write_bytes_requested"] += requested
            self._record["write_bytes_unknown_operations"] += 1
            self._manager._totals["write_bytes_unknown_operations"] += 1
            self._fail_operation("writelines", error, unknown=True)
            raise

    def seek(self, offset, whence=os.SEEK_SET):
        self._manager._check_open(self._record)
        self._record["seek_calls_attempted"] += 1
        try:
            value = self._stream.seek(offset, whence)
            self._record["seek_calls_returned"] += 1
            self._manager._totals["seek_calls_returned"] += 1
            return value
        except BaseException as error:
            self._fail_operation("seek", error)
            raise

    def tell(self):
        self._manager._check_open(self._record)
        try:
            return self._stream.tell()
        except BaseException as error:
            self._fail_operation("tell", error)
            raise

    def flush(self):
        self._manager._check_open(self._record)
        try:
            return self._stream.flush()
        except BaseException as error:
            self._fail_operation("flush", error, unknown=True)
            raise

    def close(self):
        if self._closed:
            return
        failure = None
        try:
            self._stream.close()
        except BaseException as error:  # noqa: BLE001 - close failures poison the result boundary
            failure = error
            self._manager._poison("measured output stream close failed", self._record, error)
        finally:
            self._closed = True
            self._manager._finish_open(
                self._record, failed=failure is not None or self._record["failed"]
            )
        if failure is not None:
            raise failure
        return

    def __enter__(self):
        self._manager._check_open(self._record)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self._manager._poison("exception left a measured file context", self._record, exc)
            self._record["failed"] = True
        self.close()
        return False


class _MeasuredBuffer:
    """Expose the buffered stream operations needed by byte-range journals."""

    def __init__(self, measured, buffer):
        self._measured = measured
        self._buffer = buffer

    def tell(self):
        self._measured._manager._check_open(self._measured._record)
        try:
            return self._buffer.tell()
        except BaseException as error:
            self._measured._fail_operation("buffer_tell", error)
            raise

    def __getattr__(self, name):
        if name in {"raw", "buffer"}:
            raise AttributeError(name)
        return getattr(self._buffer, name)


class OutputIO:
    """Context-managed Path.open meter constrained to one output directory."""

    def __init__(self, directory, *, max_records=270_000):
        if type(max_records) is not int or max_records < 2:
            raise ValueError("max_records must be an integer of at least two")
        self.requested_directory = Path(directory).absolute()
        self.directory = self.requested_directory.resolve(strict=True)
        if self.requested_directory.is_symlink() or self._is_junction(self.requested_directory):
            raise PermissionError("output directory cannot be a link or junction")
        if not self.directory.is_dir():
            raise NotADirectoryError(self.directory)
        self.max_records = max_records
        self.journal_path = self.directory / "output_io.jsonl"
        self._path_class = type(Path())
        self._original_open = None
        self._original_open_owned = False
        self._original_open_slot = None
        self._original_fsync = None
        self._journal = None
        self._entered = False
        self._exited = False
        self.poisoned = False
        self._lock = threading.RLock()
        self._next_open_id = 0
        self._transition_count = 0
        self._active = {}
        self._fd_handles = {}
        self._totals = Counter()
        self._unknown = Counter()
        self._first_error = None
        self._context_error = None
        self._journal_durably_synced = False

    @staticmethod
    def _is_junction(path):
        return bool(hasattr(path, "is_junction") and path.is_junction())

    def __enter__(self):
        with self._lock:
            if self._entered or self._exited:
                raise RuntimeError("OutputIO instances cannot be entered more than once")
            if (
                self.journal_path.exists()
                or self.journal_path.is_symlink()
                or self._is_junction(self.journal_path)
            ):
                raise FileExistsError(self.journal_path)
            self._journal = self._original_path_open(
                self.journal_path, "x", encoding="utf-8", newline="\n"
            )
            self._original_open = self._path_class.open
            self._original_open_owned = "open" in self._path_class.__dict__
            self._original_open_slot = self._path_class.__dict__.get("open")
            self._original_fsync = os.fsync
            manager = self

            def measured_open(path_obj, *args, **kwargs):
                return manager._open(path_obj, *args, **kwargs)

            def measured_fsync(file_descriptor):
                record = manager._fd_handles.get(file_descriptor)
                if record is None:
                    return manager._original_fsync(file_descriptor)
                record["fsync_calls_attempted"] += 1
                if manager.poisoned:
                    record["fsync_calls_failed"] += 1
                    error = OutputIOError("output I/O meter is poisoned")
                    manager._poison("fsync rejected after output I/O poison", record, error)
                    raise error
                try:
                    value = manager._original_fsync(file_descriptor)
                except BaseException as error:
                    record["fsync_calls_failed"] += 1
                    record["write_durability_unknown_operations"] += 1
                    record["failed"] = True
                    manager._unknown["fsync"] += 1
                    manager._poison("measured fsync failed", record, error)
                    raise
                record["fsync_calls_returned"] += 1
                manager._totals["fsync_calls_returned"] += 1
                return value

            self._path_class.open = measured_open
            os.fsync = measured_fsync
            self._entered = True
            return self

    @staticmethod
    def _original_path_open(path, *args, **kwargs):
        return Path.open(path, *args, **kwargs)

    def _inside(self, path_obj):
        lexical = Path(os.path.abspath(os.fspath(path_obj)))
        try:
            relative = lexical.relative_to(self.directory)
        except ValueError:
            return None
        if not relative.parts:
            raise IsADirectoryError(lexical)
        current = self.directory
        for part in relative.parts:
            current = current / part
            if current.is_symlink() or self._is_junction(current):
                raise PermissionError("measured output path cannot traverse a link or junction")
        try:
            resolved = lexical.resolve(strict=False)
            resolved.relative_to(self.directory)
        except (OSError, ValueError) as error:
            raise PermissionError("measured output path escapes its run directory") from error
        return relative.as_posix()

    @staticmethod
    def _mode(args, kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        return mode if isinstance(mode, str) else str(mode)

    def _append(self, record):
        if self._transition_count >= self.max_records:
            self._poison("output I/O transition cap exceeded", record)
            raise OutputIOError(
                "output I/O transition cap exceeded", partial_record=self.snapshot()
            )
        self._transition_count += 1
        try:
            self._journal.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )
            self._journal.flush()
            return
        except BaseException as error:
            self._poison("output I/O journal write failed", record, error)
            raise OutputIOError(
                "output I/O journal write failed", partial_record=self.snapshot(), cause=error
            ) from error

    def _open(self, path_obj, *args, **kwargs):
        with self._lock:
            try:
                lexical = Path(os.path.abspath(os.fspath(path_obj)))
                relative = lexical.relative_to(self.directory)
            except (TypeError, ValueError, OSError):
                return self._original_open(path_obj, *args, **kwargs)
            relative_name = relative.as_posix()
            if relative_name == "output_io.jsonl":
                return self._original_open(path_obj, *args, **kwargs)
            open_id = self._next_open_id
            self._next_open_id += 1
            record = self._new_handle_record(
                open_id, relative_name, self._mode(args, kwargs), kwargs.get("newline")
            )
            self._append(
                {
                    "open_id": open_id,
                    "path": relative_name,
                    "mode": record["mode"],
                    "transition": "attempted",
                }
            )
            self._totals["open_attempted"] += 1
            try:
                self._inside(path_obj)
            except BaseException as error:
                record["failed"] = True
                self._poison("unsafe path rejected by output I/O meter", record, error)
                self._finish_open(record, failed=True, opened=False)
                raise OutputIOError(
                    "unsafe path rejected by output I/O meter", partial_record=record, cause=error
                ) from error
            if self.poisoned:
                error = OutputIOError("output I/O meter is poisoned")
                self._poison("run directory operation rejected after poison", record, error)
                record["failed"] = True
                self._finish_open(record, failed=True, opened=False)
                raise OutputIOError(
                    "run directory operation rejected after poison",
                    partial_record=record,
                    cause=error,
                ) from error
            try:
                stream = self._original_open(path_obj, *args, **kwargs)
            except BaseException as error:
                record["failed"] = True
                self._poison("measured Path.open failed", record, error)
                self._finish_open(record, failed=True, opened=False)
                raise
            try:
                file_descriptor = stream.fileno()
            except BaseException as error:
                try:
                    stream.close()
                finally:
                    record["failed"] = True
                    self._poison("measured file did not expose a file descriptor", record, error)
                    self._finish_open(record, failed=True, opened=True)
                raise OutputIOError(
                    "measured file did not expose a file descriptor",
                    partial_record=record,
                    cause=error,
                ) from error
            self._fd_handles[file_descriptor] = record
            measured = _MeasuredStream(self, stream, record)
            self._active[open_id] = measured
            self._totals["open_returned"] += 1
            return measured

    @staticmethod
    def _new_handle_record(open_id, relative, mode, newline):
        return {
            "open_id": open_id,
            "path": relative,
            "mode": mode,
            "newline": newline,
            "read_calls_attempted": 0,
            "read_calls_returned": 0,
            "read_requested_units": 0,
            "read_unbounded_calls": 0,
            "read_bytes_returned": 0,
            "read_bytes_unknown_operations": 0,
            "short_reads": 0,
            "write_calls_attempted": 0,
            "write_calls_returned": 0,
            "write_bytes_requested": 0,
            "write_bytes_returned": 0,
            "write_bytes_unknown_operations": 0,
            "short_writes": 0,
            "seek_calls_attempted": 0,
            "seek_calls_returned": 0,
            "fsync_calls_attempted": 0,
            "fsync_calls_returned": 0,
            "fsync_calls_failed": 0,
            "write_durability_unknown_operations": 0,
            "failed": False,
        }

    def _check_open(self, record):
        if self.poisoned:
            record["failed"] = True
            error = OutputIOError("output I/O meter is poisoned")
            self._poison("measured handle operation rejected after poison", record, error)
            raise error
        if self._exited or record["open_id"] not in self._active:
            error = OutputIOError("measured handle is not active")
            self._poison("measured handle used outside active context", record, error)
            raise error

    def _operation_failed(self, record, operation, error, *, unknown=False):
        record["failed"] = True
        if unknown:
            self._unknown[operation] += 1
        self._poison(f"measured {operation} failed", record, error)

    def _poison(self, message, record=None, cause=None):
        with self._lock:
            self.poisoned = True
            if self._first_error is None:
                self._first_error = {
                    "message": str(message),
                    "type": type(cause).__name__ if cause is not None else "OutputIOError",
                    "cause_message": str(cause) if cause is not None else None,
                    "open_id": record.get("open_id") if isinstance(record, dict) else None,
                    "path": record.get("path") if isinstance(record, dict) else None,
                }

    def _finish_open(self, record, *, failed, opened=True):
        open_id = record["open_id"]
        self._active.pop(open_id, None)
        for file_descriptor, active_record in list(self._fd_handles.items()):
            if active_record is record:
                self._fd_handles.pop(file_descriptor, None)
        if failed:
            self._totals["open_failed"] += 1
        else:
            self._totals["open_closed_returned"] += 1
        terminal = {key: value for key, value in record.items() if key != "newline"}
        terminal.update({"transition": "failed" if failed else "returned", "opened": opened})
        self._append(terminal)

    def snapshot(self):
        with self._lock:
            return {
                "directory": self.directory.as_posix(),
                "journal": "output_io.jsonl",
                "max_records": self.max_records,
                "transition_records": self._transition_count,
                "records": self._transition_count,
                "open_attempted": self._totals["open_attempted"],
                "open_returned": self._totals["open_returned"],
                "open_failed": self._totals["open_failed"],
                "open_closed_returned": self._totals["open_closed_returned"],
                "read_calls_returned": self._totals["read_calls_returned"],
                "read_bytes_returned": self._totals["read_bytes_returned"],
                "read_bytes_unknown_operations": self._totals["read_bytes_unknown_operations"],
                "write_calls_returned": self._totals["write_calls_returned"],
                "write_bytes_returned": self._totals["write_bytes_returned"],
                "write_bytes_unknown_operations": self._totals["write_bytes_unknown_operations"],
                "seek_calls_returned": self._totals["seek_calls_returned"],
                "fsync_calls_returned": self._totals["fsync_calls_returned"],
                "unknown_operations": dict(self._unknown),
                "active_open_ids": sorted(self._active),
                "poisoned": self.poisoned,
                "first_error": self._first_error,
                "context_error": self._context_error,
                "journal_durably_synced": self._journal_durably_synced,
                "journal_closed": bool(self._journal is not None and self._journal.closed),
            }

    def __exit__(self, exc_type, exc, tb):
        close_error = None
        if exc is not None:
            self._context_error = {"type": exc_type.__name__, "message": str(exc)}
            self._poison("exception left output I/O context", cause=exc)
        try:
            for measured in list(self._active.values()):
                try:
                    measured.close()
                except BaseException as error:  # noqa: BLE001 - context cleanup must finish
                    if close_error is None:
                        close_error = error
            if self._journal is not None and not self._journal.closed:
                try:
                    self._journal.flush()
                    self._original_fsync(self._journal.fileno())
                    self._journal_durably_synced = True
                except BaseException as error:  # noqa: BLE001 - journal cleanup must poison
                    self._poison("output I/O journal close failed", cause=error)
                    if close_error is None:
                        close_error = error
                finally:
                    try:
                        self._journal.close()
                    except BaseException as error:  # noqa: BLE001 - restore hooks even on failed close
                        self._poison("output I/O journal resource close failed", cause=error)
                        if close_error is None:
                            close_error = error
        finally:
            if self._original_open is not None:
                if self._original_open_owned:
                    self._path_class.open = self._original_open_slot
                else:
                    delattr(self._path_class, "open")
            if self._original_fsync is not None:
                os.fsync = self._original_fsync
            self._exited = True
        if close_error is not None:
            raise OutputIOError(
                "output I/O cleanup failed", partial_record=self.snapshot(), cause=close_error
            ) from close_error
        return False
