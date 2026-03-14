import subprocess
import sys
import json
import os


REFERENCE_IMAGE = "data/input/viratkohli.avif"
TEST_IMAGE = "data/input/ms_dhoni.jpeg"

DERIVATIONS_DIR = "data/derivations"

def run_step(description, command):
    print("\n==============================")
    print(f"STEP: {description}")
    print("==============================\n")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\n❌ Error during: {description}")
        sys.exit(1)

    print(f"\n✅ Completed: {description}")


def show_validation_summary():

    print("\n==============================")
    print("VALIDATION SUMMARY")
    print("==============================\n")

    total = 0
    passed = 0
    failed = 0

    for file in os.listdir(DERIVATIONS_DIR):
        if file.endswith("_validation.json"):
            total += 1

            path = os.path.join(DERIVATIONS_DIR, file)

            with open(path) as f:
                data = json.load(f)

            if data["result"] == "PASS":
                passed += 1
            else:
                failed += 1

    print(f"Total images tested : {total}")
    print(f"Passed              : {passed}")
    print(f"Failed              : {failed}")

    print()

    if failed == 0:
        print("🎉 IDENTITY VALIDATION SUCCESSFUL")
    else:
        print("❌ IDENTITY VALIDATION FAILED")


def main():

    run_step(
        "Normalize Input Face",
        "python scripts/test_align.py"
    )

    run_step(
        "Create Identity Anchor",
        "python scripts/create_identity_anchor.py"
    )

    run_step(
        "Generate Identity Derivations",
        "python scripts/generate_derivations.py"
    )

    run_step(
        "Validate Identity Consistency",
        "python scripts/validate_derivations.py"
    )

    run_step(
        "Generate Identity Specification",
        "python scripts/generate_identity_spec.py"
    )

    run_step(
    "Face Recognition Check",
    f"python -m src.validation.validate_image {REFERENCE_IMAGE} {TEST_IMAGE}"
    )

    show_validation_summary()


if __name__ == "__main__":
    main()