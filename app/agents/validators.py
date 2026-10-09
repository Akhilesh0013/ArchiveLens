from app.schema import MarriageRecord

REQUIRED_FIELDS = [
    "groom.full_name",
    "bride.full_name",
]


def _get_nested(record: MarriageRecord, dotted: str):
    obj = record
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def validate_record(record: MarriageRecord) -> list[str]:
    """Return validation errors for the draft record."""
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if not _get_nested(record, field):
            errors.append(f"missing required field: {field}")

    for party in ("groom", "bride"):
        age = getattr(record, party).age
        if age is not None and not (14 <= age <= 120):
            errors.append(f"{party}.age out of range: {age}")

    if record.marriage_year is not None:
        if not (1900 <= record.marriage_year <= 1950):
            errors.append(f"marriage_year out of range: {record.marriage_year}")

    if record.marriage_month and record.marriage_day is None:
        errors.append("marriage_day missing while marriage_month is set")

    for party in ("groom", "bride"):
        name = getattr(record, party).full_name or ""
        if len(name) < 3:
            errors.append(f"{party}.full_name too short")
        if any(ch in name for ch in ['"', "*", "#"]):
            errors.append(f"{party}.full_name contains OCR noise")

    return errors
