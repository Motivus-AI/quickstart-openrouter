from __future__ import annotations

import uuid
from typing import Iterator

from addresses.models import Address


class AddressStore:
    """In-memory address book."""

    def __init__(self) -> None:
        self._by_id: dict[str, Address] = {}

    def add(self, address: Address) -> Address:
        if not address.id:
            address = address.model_copy(update={"id": str(uuid.uuid4())[:8]})
        self._by_id[address.id] = address
        return address

    def get(self, address_id: str) -> Address | None:
        return self._by_id.get(address_id)

    def list_all(self) -> list[Address]:
        return list(self._by_id.values())

    def search(self, query: str) -> list[Address]:
        q = query.lower()
        return [
            a
            for a in self._by_id.values()
            if q in a.label.lower()
            or q in a.city.lower()
            or q in a.street.lower()
            or q in a.postal_code
        ]

    def delete(self, address_id: str) -> bool:
        return self._by_id.pop(address_id, None) is not None

    def __iter__(self) -> Iterator[Address]:
        return iter(self._by_id.values())
