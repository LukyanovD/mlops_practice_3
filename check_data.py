"""check_data.py — валидация CSV, созданных prepare_data.py.

Проверяет:
  - наличие обязательного data/iris.csv;
  - отсутствие безымянных столбцов индекса (Unnamed: ...);
  - что число строк совпадает с исходным датасетом scikit-learn;
  - что базовые столбцы Iris идут первыми в исходном порядке;
  - что для iris_features значения sepal_area/petal_area равны
    произведению соответствующих исходных размеров;
  - если присутствует wine.csv — что он соответствует исходному Wine.

Выводит для каждого CSV: путь, SHA-256, rows, cols, columns, missing.
Завершается с ненулевым кодом при нарушении любого из условий.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine

DATA_DIR = Path("data")
IRIS_PATH = DATA_DIR / "iris.csv"
WINE_PATH = DATA_DIR / "wine.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def describe(path: Path, frame: pd.DataFrame) -> None:
    print(f"--- {path} ---")
    print(f"  sha256  : {sha256_file(path)}")
    print(f"  rows    : {len(frame)}")
    print(f"  columns : {len(frame.columns)}")
    print(f"  names   : {list(frame.columns)}")
    print(f"  missing : {int(frame.isna().sum().sum())}")


def validate_iris(frame: pd.DataFrame) -> None:
    ref = load_iris(as_frame=True).frame
    base_cols = list(ref.columns)  # 4 признака + target, 5 столбцов

    # 1) число строк совпадает с исходным dataset
    if len(frame) != len(ref):
        fail(f"iris.csv: rows={len(frame)}, expected {len(ref)}")

    # 2) базовые столбцы идут первыми в исходном порядке
    if list(frame.columns[: len(base_cols)]) != base_cols:
        fail(
            "iris.csv: base columns must be first and in original order, "
            f"got {list(frame.columns)}"
        )

    # 3) дополнительные столбцы: либо их нет, либо это ровно area-пара
    extra = list(frame.columns[len(base_cols):])
    expected_extra = {"sepal_area", "petal_area"}
    if set(extra) == expected_extra:
        exp_sepal = frame["sepal length (cm)"] * frame["sepal width (cm)"]
        exp_petal = frame["petal length (cm)"] * frame["petal width (cm)"]
        if not np.allclose(frame["sepal_area"], exp_sepal, rtol=1e-9, atol=1e-9):
            fail("iris.csv: sepal_area values are wrong")
        if not np.allclose(frame["petal_area"], exp_petal, rtol=1e-9, atol=1e-9):
            fail("iris.csv: petal_area values are wrong")
    elif extra:
        fail(f"iris.csv: unexpected extra columns {extra}")


def validate_wine(frame: pd.DataFrame) -> None:
    ref = load_wine(as_frame=True).frame
    if len(frame) != len(ref):
        fail(f"wine.csv: rows={len(frame)}, expected {len(ref)}")
    if list(frame.columns) != list(ref.columns):
        fail(
            "wine.csv: columns differ from sklearn Wine; "
            f"got {list(frame.columns)}, expected {list(ref.columns)}"
        )


def main() -> None:
    # "отсутствует ожидаемый файл" -> iris.csv обязателен всегда
    if not IRIS_PATH.exists():
        fail(f"expected file is missing: {IRIS_PATH}")

    files = [IRIS_PATH]
    if WINE_PATH.exists():
        files.append(WINE_PATH)

    for path in files:
        frame = pd.read_csv(path)
        describe(path, frame)

        # безымянный столбец индекса (например, "Unnamed: 0")
        unnamed = [c for c in frame.columns if str(c).startswith("Unnamed:")]
        if unnamed:
            fail(f"{path}: unnamed index-like columns: {unnamed}")

        if path == IRIS_PATH:
            validate_iris(frame)
        elif path == WINE_PATH:
            validate_wine(frame)

    print("\nOK: all checks passed")


if __name__ == "__main__":
    main()