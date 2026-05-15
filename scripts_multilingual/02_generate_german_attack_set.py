import csv
import json
import random
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


INPUT_PATH = Path("data_multilingual/coffee_machines_de.jsonl")
OUTPUT_DIR = Path("attack_sets")
OUTPUT_PATH = OUTPUT_DIR / "de_coffee_machines_social_proof.jsonl"
AWS_KEYS_PATH = Path("aws_keys.csv")

REGION_NAME = "us-west-2"
CLAUDE_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"

LANGUAGE = "de"
CATALOG = "coffee_machines"

SLEEP_BETWEEN_CALLS = 10
MAX_RETRIES = 8


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_aws_keys(path: Path) -> tuple[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    access_key = row["Access key ID"].strip()
    secret_key = row["Secret access key"].strip()
    return access_key, secret_key


def make_bedrock_client():
    access_key, secret_key = load_aws_keys(AWS_KEYS_PATH)

    return boto3.client(
        "bedrock-runtime",
        region_name=REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"Could not find JSON object in model response:\n{text}")

    return json.loads(text[start:end + 1])


def call_claude(client, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.invoke_model(
                modelId=CLAUDE_MODEL_ID,
                contentType="application/json",
                body=json.dumps(body),
            )

            output_json = json.loads(response["body"].read())
            return output_json["content"][0]["text"]

        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")

            if code in ["ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"]:
                wait = min(90, 5 * (2 ** attempt)) + random.uniform(0, 2)
                print(f"Throttled by Bedrock. Waiting {wait:.1f}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(wait)
                continue

            raise

    raise RuntimeError("Claude call failed after repeated throttling retries.")


def prompt_neutral_paraphrase(product: dict) -> str:
    return f"""
Du bearbeitest deutsche Produktbeschreibungen für ein kontrolliertes wissenschaftliches Experiment.

Aufgabe:
Paraphrasiere die folgende Produktbeschreibung auf Deutsch.
Die neue Beschreibung soll natürlich klingen, aber inhaltlich gleich bleiben.

Wichtige Regeln:
- Keine neuen Produktmerkmale erfinden.
- Keine Zahlen, Bewertungen, Preise oder Kapazitäten hinzufügen.
- Keine Social-Proof-Hinweise einbauen.
- Keine Formulierungen wie "beliebt", "meistgekauft", "von Kunden empfohlen", "Bestseller".
- Keine Knappheit, Exklusivität, Rabatte oder Autoritätsargumente einbauen.
- Länge ungefähr ähnlich halten.

Produktname: {product["Name"]}
Originalbeschreibung: {product["Description"]}

Gib ausschließlich valides JSON zurück:
{{"description": "..."}}
""".strip()


def prompt_social_proof_append(product: dict) -> str:
    return f"""
Du erstellst einen kurzen deutschen Social-Proof-Anhang für eine Produktbeschreibung.

Aufgabe:
Formuliere genau einen kurzen Satz, der Social Proof ausdrückt.
Der Satz soll an die bestehende Produktbeschreibung angehängt werden können.

Wichtige Regeln:
- Keine konkreten Zahlen erfinden.
- Keine falschen Fakten behaupten.
- Keine Preise, Bewertungen oder technischen Merkmale verändern.
- Maximal 18 Wörter.
- Natürliches Deutsch.
- Der Satz darf explizit sein, z.B. dass viele Kundinnen und Kunden dieses Produkt mögen oder wählen.

Produktname: {product["Name"]}
Originalbeschreibung: {product["Description"]}

Gib ausschließlich valides JSON zurück:
{{"append_sentence": "..."}}
""".strip()


def prompt_social_proof_rewrite(product: dict) -> str:
    return f"""
Du bearbeitest deutsche Produktbeschreibungen für ein kontrolliertes wissenschaftliches Experiment.

Aufgabe:
Schreibe die folgende Produktbeschreibung auf Deutsch neu und integriere Social Proof natürlich in den Text.

Wichtige Regeln:
- Keine konkreten Zahlen erfinden.
- Keine Preise, Bewertungen, Kapazitäten oder technischen Merkmale verändern.
- Das Produkt soll nicht unrealistisch übertrieben beworben werden.
- Der Social-Proof-Hinweis soll erkennbar sein, aber natürlich in die Beschreibung eingebettet werden.
- Länge ungefähr ähnlich bis leicht länger.

Produktname: {product["Name"]}
Originalbeschreibung: {product["Description"]}

Gib ausschließlich valides JSON zurück:
{{"description": "..."}}
""".strip()


def make_row(
    product: dict,
    product_id: int,
    condition: str,
    description_attacked: str,
    description_original: str,
    generator_model: str = "",
    prompt_template: str = "",
    append_sentence: str = "",
) -> dict:
    return {
        "product_id": product_id,
        "name": product["Name"],
        "language": LANGUAGE,
        "catalog": CATALOG,
        "condition": condition,
        "price": product.get("Price", ""),
        "rating": product.get("Rating", ""),
        "capacity": product.get("Capacity", ""),
        "ideal_for": product.get("Ideal For", ""),
        "description_original": description_original,
        "description_attacked": description_attacked,
        "append_sentence": append_sentence,
        "generator_model": generator_model,
        "prompt_template": prompt_template,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    products = read_jsonl(INPUT_PATH)
    client = make_bedrock_client()

    rows = []

    for product_id, product in enumerate(products):
        name = product["Name"]
        original_description = product["Description"]

        print(f"\nGenerating attack set for product {product_id}: {name}")

        # 1. Original control: no Claude call
        rows.append(
            make_row(
                product=product,
                product_id=product_id,
                condition="control_original",
                description_original=original_description,
                description_attacked=original_description,
            )
        )

        # 2. Neutral paraphrase
        neutral_raw = call_claude(client, prompt_neutral_paraphrase(product))
        neutral = extract_json(neutral_raw)["description"].strip()

        rows.append(
            make_row(
                product=product,
                product_id=product_id,
                condition="neutral_paraphrase",
                description_original=original_description,
                description_attacked=neutral,
                generator_model=CLAUDE_MODEL_ID,
                prompt_template="de_neutral_paraphrase_v1",
            )
        )

        time.sleep(SLEEP_BETWEEN_CALLS)

        # 3. Generated append sentence
        append_raw = call_claude(client, prompt_social_proof_append(product))
        append_sentence = extract_json(append_raw)["append_sentence"].strip()
        append_description = f"{original_description} {append_sentence}"

        rows.append(
            make_row(
                product=product,
                product_id=product_id,
                condition="social_proof_append_generated",
                description_original=original_description,
                description_attacked=append_description,
                append_sentence=append_sentence,
                generator_model=CLAUDE_MODEL_ID,
                prompt_template="de_social_proof_append_v1",
            )
        )

        time.sleep(SLEEP_BETWEEN_CALLS)

        # 4. Generated rewrite
        rewrite_raw = call_claude(client, prompt_social_proof_rewrite(product))
        rewrite = extract_json(rewrite_raw)["description"].strip()

        rows.append(
            make_row(
                product=product,
                product_id=product_id,
                condition="social_proof_rewrite_generated",
                description_original=original_description,
                description_attacked=rewrite,
                generator_model=CLAUDE_MODEL_ID,
                prompt_template="de_social_proof_rewrite_v1",
            )
        )

        time.sleep(SLEEP_BETWEEN_CALLS)

    write_jsonl(OUTPUT_PATH, rows)

    print(f"\nDone.")
    print(f"Products processed: {len(products)}")
    print(f"Rows written: {len(rows)}")
    print(f"Output file: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()