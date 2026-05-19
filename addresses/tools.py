from __future__ import annotations

from langchain.tools import tool

from addresses.models import Address
from addresses.store import AddressStore


def create_address_tools(store: AddressStore) -> list:
    @tool
    def add_address(
        label: str,
        street: str,
        city: str,
        number: str = "",
        state: str = "",
        postal_code: str = "",
        country: str = "US",
        notes: str = "",
    ) -> str:
        """Save a new address in the address book."""
        addr = store.add(
            Address(
                label=label,
                street=street,
                number=number,
                city=city,
                state=state,
                postal_code=postal_code,
                country=country,
                notes=notes,
            )
        )
        return f"Saved address id={addr.id} ({addr.label})"

    @tool
    def list_addresses() -> str:
        """List all saved addresses with ids."""
        items = store.list_all()
        if not items:
            return "No addresses saved."
        return "\n".join(
            f"- id={a.id} | {a.label}: {a.street}"
            f"{f' {a.number}' if a.number else ''}, {a.city}"
            for a in items
        )

    @tool
    def search_addresses(query: str) -> str:
        """Search by label, street, city, or postal code."""
        hits = store.search(query)
        if not hits:
            return f"No addresses found for '{query}'."
        return "\n".join(
            f"id={a.id} | {a.label}\n{a.mailing_format()}" for a in hits
        )

    @tool
    def format_address(address_id: str) -> str:
        """Return mailing format for an address by id."""
        addr = store.get(address_id)
        if addr is None:
            return f"No address with id={address_id}"
        return addr.mailing_format()

    @tool
    def delete_address(address_id: str) -> str:
        """Delete an address by id."""
        if store.delete(address_id):
            return f"Deleted address {address_id}."
        return f"No address with id={address_id}"

    return [add_address, list_addresses, search_addresses, format_address, delete_address]
