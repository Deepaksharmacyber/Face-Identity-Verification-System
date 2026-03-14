Run normalization
python scripts/test_align.py

Create embedding
python src/reference/create_identity_anchor.py

Generate derivations
python scripts/generate_derivations.py

Validate identity
python scripts/validate_derivations.py

Generate spec
python scripts/generate_identity_spec.py