from __future__ import annotations

from pydantic import BaseModel, Field


class Address(BaseModel):
    """Normalized postal address."""

    id: str | None = None
    label: str = Field(description="Short name, e.g. Home, Office")
    street: str
    number: str = ""
    city: str
    state: str = ""
    postal_code: str = ""
    country: str = "US"
    notes: str = ""

    def mailing_format(self) -> str:
        line1 = f"{self.street} {self.number}".strip() if self.number else self.street
        line2 = ", ".join(
            part for part in (self.postal_code, self.city, self.state) if part
        )
        lines = [line1]
        if line2:
            lines.append(line2)
        if self.country:
            lines.append(self.country)
        return "\n".join(lines)
