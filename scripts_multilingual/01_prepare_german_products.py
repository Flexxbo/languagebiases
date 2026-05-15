import json
from pathlib import Path


INPUT_PATH = Path("data/coffee_machines.jsonl")
OUTPUT_DIR = Path("data_multilingual")
OUTPUT_PATH = OUTPUT_DIR / "coffee_machines_de.jsonl"


GERMAN_DESCRIPTIONS = {
    "FrenchPress Classic": "Traditionelle French Press für eine reichhaltige und aromatische Tasse Kaffee.",
    "SingleServe Wonder": "Kompakte und praktische Einzelportions-Kaffeemaschine für eine schnelle Tasse Kaffee.",
    "QuickBrew Express": "Schnelle und effiziente Kaffeemaschine für eine zügig zubereitete Tasse Kaffee.",
    "BrewMaster Classic": "Robuste und einfach zu bedienende Kaffeemaschine mit zeitlosem Design.",
    "ColdBrew Master": "Spezialisierte Maschine zur Zubereitung von mildem und erfrischendem Cold Brew Kaffee.",
    "Grind&Brew Plus": "Kaffeemaschine mit integriertem Mahlwerk für frisch gemahlenen Kaffee bei jeder Zubereitung.",
    "EspressoMaster 2000": "Kompakte und effiziente Espressomaschine mit fortschrittlicher Brütechnologie.",
    "LatteArt Pro": "Fortschrittliche Kaffeemaschine mit integriertem Milchaufschäumer für perfekte Lattes und Cappuccinos.",
    "Cappuccino King": "Hochwertige Maschine zur Zubereitung von Cappuccinos in professioneller Qualität.",
    "CafePro Elite": "Professionelle Kaffeemaschine mit mehreren Brühoptionen und elegantem Design.",
}


GERMAN_IDEAL_FOR = {
    "French press enthusiasts": "Liebhaberinnen und Liebhaber von French-Press-Kaffee",
    "Individuals on-the-go": "Menschen, die viel unterwegs sind",
    "Busy individuals": "Beschäftigte Personen",
    "Home use": "Nutzung zu Hause",
    "Cold brew lovers": "Cold-Brew-Liebhaberinnen und -Liebhaber",
    "Coffee purists": "Kaffeepuristinnen und -puristen",
    "Espresso enthusiasts": "Espresso-Liebhaberinnen und -Liebhaber",
    "Latte and cappuccino lovers": "Latte- und Cappuccino-Liebhaberinnen und -Liebhaber",
    "Cappuccino aficionados": "Cappuccino-Kennerinnen und -Kenner",
    "Coffee enthusiasts and small cafes": "Kaffeeliebhaberinnen und -liebhaber sowie kleine Cafés",
}


def read_jsonl(path: Path) -> list[dict]:
    products = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    return products


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def translate_product(product: dict) -> dict:
    product_de = dict(product)
    name = product_de.get("Name")

    if name not in GERMAN_DESCRIPTIONS:
        raise ValueError(f"No German description found for product name: {name}")

    product_de["Description"] = GERMAN_DESCRIPTIONS[name]
    product_de["Language"] = "de"
    product_de["SourceLanguage"] = "en"

    if "Ideal For" in product_de:
        original_ideal_for = product_de["Ideal For"]
        product_de["Ideal For"] = GERMAN_IDEAL_FOR.get(original_ideal_for, original_ideal_for)

    if "Capacity" in product_de:
        product_de["Capacity"] = str(product_de["Capacity"]).replace("cups", "Tassen").replace("cup", "Tasse")

    return product_de


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    products_en = read_jsonl(INPUT_PATH)
    products_de = [translate_product(product) for product in products_en]

    write_jsonl(OUTPUT_PATH, products_de)

    print(f"Read {len(products_en)} products from: {INPUT_PATH}")
    print(f"Wrote German product file to: {OUTPUT_PATH}")

    print("\nPreview:")
    for product in products_de:
        print(f"- {product['Name']}: {product['Description']}")


if __name__ == "__main__":
    main()