import pytest

from momba_engine import zones


def test_includes() -> None:
    x = zones.Zone.new_i64(4)
    y = zones.Zone.new_i64(4)
    print(x, y)
    assert x.includes(y)


def test_include_different_types() -> None:
    x = zones.Zone.new_i64(4)
    y = zones.Zone.new_f64(4)
    with pytest.raises(ValueError):
        x.includes(y)


def test_include_different_num_variables() -> None:
    x = zones.Zone.new_i64(2)
    y = zones.Zone.new_i64(4)
    with pytest.raises(ValueError):
        x.includes(y)
