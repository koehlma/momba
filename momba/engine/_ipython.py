from . import explore


def state_repr_html(state: explore.State[explore.TimeType]) -> str:
    return "<strong>Hello!</strong>"


explore.State._repr_html_ = state_repr_html  # type: ignore
