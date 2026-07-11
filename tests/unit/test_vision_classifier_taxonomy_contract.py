from models.vision.classifier import WasteClassifier


def test_classifier_primary_material_bin_uses_canonical_special_routes():
    classifier = WasteClassifier(device="cpu")

    material, bin_type = classifier._canonical_material_bin(
        "plastic_shopping_bags",
        fallback_material="LDPE",
        fallback_bin="recycle",
    )

    assert material == "LDPE"
    assert bin_type == "special"


def test_classifier_primary_material_bin_uses_canonical_donation_routes():
    classifier = WasteClassifier(device="cpu")

    material, bin_type = classifier._canonical_material_bin(
        "clothing",
        fallback_material="cotton",
        fallback_bin="landfill",
    )

    assert material == "textile"
    assert bin_type == "donate"


def test_classifier_unknown_item_keeps_legacy_fallback():
    classifier = WasteClassifier(device="cpu")

    material, bin_type = classifier._canonical_material_bin(
        "unknown_future_item",
        fallback_material="other",
        fallback_bin="hazardous",
    )

    assert material == "other"
    assert bin_type == "hazardous"
