
from src.data import load_wine_data, split_data


def test_wine_dataset_feature_count():
    X, y = load_wine_data()

    assert X.shape[1] == 13
    assert len(X) == 178
    assert len(y) == 178


def test_wine_dataset_has_no_nulls():
    X, y = load_wine_data()

    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_wine_dataset_has_three_classes():
    X, y = load_wine_data()

    assert set(y.unique()) == {0, 1, 2}


def test_stratified_train_test_split():
    X, y = load_wine_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) == 142
    assert len(X_test) == 36

    assert len(y_train) == 142
    assert len(y_test) == 36

    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}
