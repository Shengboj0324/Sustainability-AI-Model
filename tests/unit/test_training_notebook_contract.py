import importlib.util
from pathlib import Path


VALIDATOR_PATH = Path("scripts/validate_training_notebook.py").resolve()
spec = importlib.util.spec_from_file_location("validate_training_notebook", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(validator)


def test_training_notebook_contract_is_current():
    notebook = Path("Sustainability_AI_Model_Training.ipynb")
    assert validator.validate_notebook(notebook) == []
