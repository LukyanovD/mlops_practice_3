import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_wine


def save_frame(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values(by=list(df.columns), kind="mergesort").reset_index(drop=True)
    df_sorted.to_csv(path_obj, index=False, lineterminator="\n", float_format="%.10g")


def main(mode: str) -> None:
    if mode == "base":
        iris = load_iris(as_frame=True).frame
        save_frame(iris, "data/iris.csv")
    elif mode == "wine":
        wine = load_wine(as_frame=True).frame
        save_frame(wine, "data/wine.csv")
    elif mode == "iris_features":
        iris = load_iris(as_frame=True).frame
        iris["sepal_area"] = iris["sepal length (cm)"] * iris["sepal width (cm)"]
        iris["petal_area"] = iris["petal length (cm)"] * iris["petal width (cm)"]
        save_frame(iris, "data/iris.csv")
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main(sys.argv[1])