<!-- @kody-sync -->
# Models — `src/xcstrings_translator/models.py`

Applies to: `src/xcstrings_translator/models.py`
Severity: high

Pydantic models mirror Apple's `.xcstrings` JSON. Round-trip fidelity is critical:

- Flag any change that could **drop unknown keys**, **reorder** keys, or otherwise
  alter the file on serialize/deserialize.
- The deserialize → serialize round-trip must reproduce the original file faithfully;
  unknown/forward-compatible fields must be preserved, not silently discarded.
