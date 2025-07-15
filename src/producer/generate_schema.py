import json
from pathlib import Path

from avro_schema import transaction_schema, user_schema, card_schema


def main():
    parent_dir = Path(__file__).resolve().parent.parent
    out_dir = parent_dir / "avro_schemas"
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_map = {
        "transaction.avsc": transaction_schema,
        "user.avsc": user_schema,
        "card.avsc": card_schema,
    }

    for filename, schema in schema_map.items():
        out_file = out_dir / filename
        with open(out_file, "w") as f:
            json.dump(schema, f, indent=2)


if __name__ == "__main__":
    main()
