"""Pydantic schemas for Baltimore City marriage license certificates (1930s–1940s).

Field layout mirrors the printed Maryland form:
  header → marriage event → groom block → bride block → consent/relationship → officiant.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _coerce_optional_int(value: Any) -> int | None:
    """Coerce LLM string outputs like ``\"20\"`` into integers."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return int(stripped)
    return value


class PartyDetails(BaseModel):
    """Person block for groom or bride on a marriage license."""

    full_name: str | None = Field(
        None, description="Full name as printed on the certificate."
    )
    residence: str | None = Field(
        None, description="City and state of residence, e.g. BALTIMORE MD."
    )
    age: int | None = Field(None, description="Age in years.")
    color: str | None = Field(
        None,
        description="Race/color code recorded on the form, e.g. WHT, COL.",
    )
    nativity: str | None = Field(
        None, description="Birth state or country, e.g. TENN, NEW YORK, MARYLAND."
    )
    occupation: str | None = Field(
        None, description="Occupation; often NONE for bride."
    )
    marital_status: str | None = Field(
        None,
        description="Marital status: single, widower, widow, or divorced.",
    )

    @field_validator("age", mode="before")
    @classmethod
    def coerce_age(cls, value: Any) -> int | None:
        return _coerce_optional_int(value)


class OfficiantDetails(BaseModel):
    """Officiating clergyman or official who performed the ceremony."""

    signature: str | None = Field(
        None, description="Signature or printed name of the officiant."
    )
    official_character: str | None = Field(
        None, description="Official title or character, if filled in."
    )
    address: str | None = Field(None, description="Officiant address, if filled in.")



# ================================================================================================================================


class MarriageRecord(BaseModel):
    """Structured extraction from one marriage license / certificate page."""

    # --- Header / identifiers ---
    jurisdiction_note: str | None = Field(
        None, description="Banner text, e.g. GOOD ONLY IN BALTIMORE CITY."
    )
    state: str | None = Field(None, description="Issuing state, e.g. MARYLAND.")
    city: str | None = Field(None, description="Issuing city, e.g. BALTIMORE CITY.")
    license_number: str | None = Field(
        None, description="License number (No. field on form)."
    )
    application_number: str | None = Field(
        None, description="Application number (APPLICATION / APPLI / APP field)."
    )

    # --- Marriage event (certification block) ---
    marriage_day: int | None = Field(
        None, description="Day of month the marriage was performed."
    )
    marriage_month: str | None = Field(
        None, description="Month name, e.g. Aug, August."
    )
    marriage_year: int | None = Field(
        None, description="Four-digit year, e.g. 1939."
    )
    marriage_year_text: str | None = Field(
        None,
        description="Verbatim year phrase, e.g. Nineteen hundred and thirty-nine.",
    )
    marriage_location: str | None = Field(
        None,
        description="Place where the ceremony was performed, e.g. Baltimore, Balto.",
    )

    groom: PartyDetails = Field(default_factory=PartyDetails)
    bride: PartyDetails = Field(default_factory=PartyDetails)


    groom_consent_by: str | None = Field(
        None,
        description="Parent or guardian consenting if groom is a minor.",
    )
    bride_consent_by: str | None = Field(
        None,
        description="Parent or guardian consenting if bride is a minor.",
    )
    parents_consent_filed: bool | None = Field(
        None,
        description="True if the form notes that parents' consent was filed.",
    )
    relationship_between_parties: str | None = Field(
        None,
        description="Whether parties are related, e.g. NOT RELATED, NONE.",
    )

    # --- Officiant ---
    officiant: OfficiantDetails = Field(default_factory=OfficiantDetails)

    # --- Provenance (added by pipeline, not printed on form) ---
    source_pdf: str | None = None
    source_page: int | None = None
    ocr_engine: str | None = None

    @field_validator("marriage_day", "marriage_year", "source_page", mode="before")
    @classmethod
    def coerce_int_fields(cls, value: Any) -> int | None:
        return _coerce_optional_int(value)


def marriage_record_to_payload(record: MarriageRecord) -> dict[str, Any]:
    """Build a plain dict from a record without Pydantic serialization.

    LangChain structured output can construct models with string values in int
    fields; dumping those instances triggers serializer warnings. Reading
    attributes directly avoids that and lets ``model_validate`` coerce types.
    """
    payload: dict[str, Any] = {}
    for name in MarriageRecord.model_fields:
        value = getattr(record, name)
        if isinstance(value, BaseModel):
            payload[name] = {
                field: getattr(value, field)
                for field in type(value).model_fields
            }
        else:
            payload[name] = value
    return payload


def normalize_marriage_record(record: MarriageRecord) -> MarriageRecord:
    """Re-parse a record so int fields are coerced and validators run."""
    return MarriageRecord.model_validate(marriage_record_to_payload(record))
