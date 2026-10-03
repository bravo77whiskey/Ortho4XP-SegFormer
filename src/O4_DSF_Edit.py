"""Byte-level edits to compiled DSF files that leave geometry untouched.

Two things need changing for tile vegetation replacement, and both live in the
small text-bearing atoms at the front of a DSF:

  * header properties  (HEAD/PROP) — to add or remove ``sim/exclude_for``
  * definition tables  (DEFN/OBJT, DEFN/POLY, DEFN/NETW) — to repoint one
    asset reference at a private, vegetation-free replacement

Everything else — the command stream, the pools, the raster layers — is copied
through verbatim, so no coordinate is ever requantized and no record is lost.
Only the enclosing atom lengths and the trailing MD5 footer are recomputed.

Format: https://developer.x-plane.com/article/dsf-file-format-specification/
Atom IDs are stored little-endian, so ``HEAD`` is ``b"DAEH"`` on disk.

Compressed (7z-packed) DSFs are rejected rather than silently mishandled; the
packages this touches ship plain files.
"""

import hashlib
import os
import struct

MAGIC = b"XPLNEDSF"
VERSION = 1
_HEADER_LEN = 12
_FOOTER_LEN = 16

# On-disk (little-endian) spellings of the atom IDs used here.
ATOM_HEAD = b"DAEH"
ATOM_PROP = b"PORP"
ATOM_DEFN = b"NFED"
ATOM_TERT = b"TRET"
ATOM_OBJT = b"TJBO"
ATOM_POLY = b"YLOP"
ATOM_NETW = b"WTEN"
ATOM_DEMN = b"NMED"

#: Definition tables addressable by name, in the order X-Plane stores them.
DEFINITION_ATOMS = {
    "terrain": ATOM_TERT,
    "object": ATOM_OBJT,
    "polygon": ATOM_POLY,
    "network": ATOM_NETW,
    "raster": ATOM_DEMN,
}

_CONTAINER_ATOMS = (ATOM_HEAD, ATOM_DEFN)


class DsfError(ValueError):
    """A DSF this module refuses to read or rewrite."""


def _walk(data, start, end):
    """Yield ``(atom_id, payload_start, payload_end, atom_end)`` over a span."""
    while start < end:
        if start + 8 > end:
            raise DsfError("truncated atom header")
        atom_id, size = struct.unpack_from("<4sI", data, start)
        if size < 8 or start + size > end:
            raise DsfError(f"atom {atom_id!r} has an invalid length {size}")
        yield atom_id, start + 8, start + size, start + size
        start += size


def _split_strings(payload):
    """Split a null-terminated string table, rejecting an unterminated one."""
    if not payload:
        return []
    if not payload.endswith(b"\0"):
        raise DsfError("string table is not null-terminated")
    return [item.decode("utf-8") for item in payload[:-1].split(b"\0")]


def _join_strings(values):
    if not values:
        return b""
    return b"".join(value.encode("utf-8") + b"\0" for value in values)


class DsfFile:
    """A parsed DSF whose atoms can be replaced individually and re-emitted."""

    def __init__(self, data):
        if len(data) < _HEADER_LEN + _FOOTER_LEN:
            raise DsfError("file is too small to be a DSF")
        if data[:8] != MAGIC:
            raise DsfError("not a DSF (missing XPLNEDSF magic)")
        version = struct.unpack_from("<I", data, 8)[0]
        if version != VERSION:
            raise DsfError(f"unsupported DSF version {version}")
        body = data[:-_FOOTER_LEN]
        if hashlib.md5(body).digest() != data[-_FOOTER_LEN:]:
            raise DsfError("MD5 footer does not match the file body")
        self._data = data
        # Top-level atoms, each as (atom_id, payload bytes). Container atoms
        # keep a parsed child list so a nested rewrite stays cheap.
        self._atoms = []
        for atom_id, payload_start, payload_end, _ in _walk(
            data, _HEADER_LEN, len(data) - _FOOTER_LEN
        ):
            payload = data[payload_start:payload_end]
            if atom_id in _CONTAINER_ATOMS:
                children = [
                    (child_id, data[child_start:child_end])
                    for child_id, child_start, child_end, _ in _walk(
                        data, payload_start, payload_end
                    )
                ]
                self._atoms.append((atom_id, children))
            else:
                self._atoms.append((atom_id, payload))
        if not any(atom_id == ATOM_HEAD for atom_id, _ in self._atoms):
            raise DsfError("no HEAD atom")

    # ── reading ──────────────────────────────────────────────────────────
    @classmethod
    def read(cls, path):
        with open(path, "rb") as handle:
            return cls(handle.read())

    def _children(self, container_id):
        for atom_id, payload in self._atoms:
            if atom_id == container_id:
                return payload
        return None

    def properties(self):
        """Return header properties as an ordered ``[(key, value), ...]`` list.

        Keys repeat in real DSFs — several ``sim/exclude_obj`` rectangles, for
        instance — so this is deliberately not a dict.
        """
        children = self._children(ATOM_HEAD) or []
        for child_id, payload in children:
            if child_id != ATOM_PROP:
                continue
            strings = _split_strings(payload)
            if len(strings) % 2:
                raise DsfError("header property table has an odd string count")
            return list(zip(strings[0::2], strings[1::2]))
        return []

    def definitions(self, kind):
        """Return the asset paths of one definition table (empty when absent)."""
        atom_id = DEFINITION_ATOMS[kind]
        children = self._children(ATOM_DEFN) or []
        for child_id, payload in children:
            if child_id == atom_id:
                return _split_strings(payload)
        return []

    def all_definitions(self):
        """Return ``{kind: [paths]}`` for every definition table present."""
        return {kind: self.definitions(kind) for kind in DEFINITION_ATOMS}

    def atom_map(self):
        """Return ``{atom_id: payload}`` for verification against another file.

        Container atoms map to ``{child_id: payload}``. Comparing two of these
        proves which records a rewrite left alone, without re-deriving them
        through the same code that produced the rewrite.
        """
        result = {}
        for atom_id, payload in self._atoms:
            if atom_id in _CONTAINER_ATOMS:
                result[atom_id] = dict(payload)
            else:
                result[atom_id] = payload
        return result

    # ── writing ──────────────────────────────────────────────────────────
    def _replace_child(self, container_id, child_id, payload):
        for index, (atom_id, children) in enumerate(self._atoms):
            if atom_id != container_id:
                continue
            updated = list(children)
            for child_index, (existing_id, _) in enumerate(updated):
                if existing_id == child_id:
                    updated[child_index] = (child_id, payload)
                    break
            else:
                updated.append((child_id, payload))
            self._atoms[index] = (atom_id, updated)
            return
        raise DsfError(f"container atom {container_id!r} is missing")

    def set_properties(self, pairs):
        """Replace the whole header property table."""
        flat = []
        for key, value in pairs:
            flat.extend((str(key), str(value)))
        self._replace_child(ATOM_HEAD, ATOM_PROP, _join_strings(flat))

    def set_definitions(self, kind, paths):
        """Replace one definition table, keeping its entry count and order.

        The command stream addresses definitions by index, so a caller may only
        substitute paths in place — never insert, drop or reorder them.
        """
        atom_id = DEFINITION_ATOMS[kind]
        current = self.definitions(kind)
        paths = list(paths)
        if len(paths) != len(current):
            raise DsfError(
                f"{kind} definition count would change "
                f"({len(current)} -> {len(paths)}); indices must stay stable"
            )
        if not current:
            return
        self._replace_child(ATOM_DEFN, atom_id, _join_strings(paths))

    def to_bytes(self):
        """Re-emit the file with recomputed atom lengths and MD5 footer."""
        chunks = [self._data[:_HEADER_LEN]]
        for atom_id, payload in self._atoms:
            if atom_id in _CONTAINER_ATOMS:
                inner = b"".join(
                    struct.pack("<4sI", child_id, len(child_payload) + 8)
                    + child_payload
                    for child_id, child_payload in payload
                )
                payload = inner
            chunks.append(struct.pack("<4sI", atom_id, len(payload) + 8))
            chunks.append(payload)
        body = b"".join(chunks)
        return body + hashlib.md5(body).digest()


def upsert_property(pairs, key, value):
    """Return ``pairs`` with exactly one ``key`` entry set to ``value``.

    A replaced key keeps its original position so the header reads the same as
    before apart from the one changed line; a new key is appended.
    """
    result = []
    replaced = False
    for existing_key, existing_value in pairs:
        if existing_key != key:
            result.append((existing_key, existing_value))
        elif not replaced:
            result.append((key, value))
            replaced = True
    if not replaced:
        result.append((key, value))
    return result


def remove_property(pairs, key, value=None):
    """Return ``pairs`` without the entries matching ``key`` (and ``value``)."""
    return [
        (existing_key, existing_value)
        for existing_key, existing_value in pairs
        if existing_key != key or (value is not None and existing_value != value)
    ]


def atomic_write(path, data, fsync=True):
    """Write ``data`` to ``path`` through a temporary file in the same folder."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.o4xp-tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# ── command stream ───────────────────────────────────────────────────────────
# Walking the commands is only ever used to *count* what a definition places,
# so the report can say how many trees a tile loses. Nothing here rewrites the
# stream: command sizes are from X-Plane/xptools src/DSF/DSFLib.cpp and an
# unknown opcode raises rather than guessing a length and desynchronising.

ATOM_CMDS = b"SDMC"

_FIXED_COMMAND_WIDTHS = {
    1: 2, 2: 4, 3: 1, 4: 2, 5: 4, 6: 1, 7: 2, 8: 4,
    10: 4, 13: 6, 16: 0, 17: 1, 18: 9, 25: 4, 28: 4, 31: 4,
}
_SET_DEFINITION_COMMANDS = (3, 4, 5)
_OBJECT_COMMANDS = (7, 8)
_NETWORK_COMMANDS = (9, 10, 11)
_POLYGON_COMMANDS = (12, 13, 14, 15)


def _iter_commands(payload):
    """Yield ``(opcode, start, end, definition)`` for every command."""
    pos = 0
    definition = None
    size = len(payload)

    def number(offset, width):
        if offset + width > size:
            raise DsfError("command field runs past the end of the atom")
        return int.from_bytes(payload[offset:offset + width], "little")

    while pos < size:
        start = pos
        opcode = payload[pos]
        pos += 1
        if opcode in _FIXED_COMMAND_WIDTHS:
            width = _FIXED_COMMAND_WIDTHS[opcode]
            if opcode in _SET_DEFINITION_COMMANDS:
                definition = number(pos, width)
            pos += width
        elif opcode in (9, 11, 23, 24, 26, 27, 29, 30):
            count = number(pos, 1)
            stride = 4 if opcode in (11, 24, 27, 30) else 2
            pos += 1 + count * stride
        elif opcode == 12:
            pos += 3 + 2 * number(pos + 2, 1)
        elif opcode == 14:
            count = number(pos + 2, 1)
            pos += 3
            for _ in range(count):
                pos += 1 + 2 * number(pos, 1)
        elif opcode == 15:
            pos += 5 + 2 * number(pos + 2, 1)
        elif opcode in (32, 33, 34):
            width = {32: 1, 33: 2, 34: 4}[opcode]
            pos += width + number(pos, width)
        else:
            raise DsfError(f"unsupported command {opcode} at offset {start}")
        if pos > size:
            raise DsfError("command extends beyond the command atom")
        yield opcode, start, pos, definition


def count_definition_usage(data):
    """Return ``{kind: {definition_index: placements}}`` for one DSF's commands.

    ``kind`` is ``object``, ``polygon`` or ``network``. An object range counts
    as the number of objects it places; every other command counts as one.
    """
    counts = {"object": {}, "polygon": {}, "network": {}}
    for atom_id, payload_start, payload_end, _ in _walk(
        data, _HEADER_LEN, len(data) - _FOOTER_LEN
    ):
        if atom_id != ATOM_CMDS:
            continue
        payload = data[payload_start:payload_end]
        for opcode, start, end, definition in _iter_commands(payload):
            if definition is None:
                continue
            if opcode in _OBJECT_COMMANDS:
                kind = "object"
                if opcode == 8:  # object range: first and last pool index
                    first = int.from_bytes(payload[start + 1:start + 3], "little")
                    last = int.from_bytes(payload[start + 3:start + 5], "little")
                    amount = max(0, last - first)
                else:
                    amount = 1
            elif opcode in _POLYGON_COMMANDS:
                kind, amount = "polygon", 1
            elif opcode in _NETWORK_COMMANDS:
                kind, amount = "network", 1
            else:
                continue
            counts[kind][definition] = counts[kind].get(definition, 0) + amount
    return counts
