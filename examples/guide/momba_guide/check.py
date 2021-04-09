# flake8: noqa


from . import model


from momba.tools import modest, storm

modest_checker = modest.get_checker(accept_license=True)
storm_checker = storm.get_checker(accept_license=True)


from momba.moml import expr, prop

properties = {
    "goal": prop(
        "min({ Pmax(F($has_won)) | initial })",
        has_won=model.has_won(expr("pos_x"), model.track),
    ),
}

results = {
    checker: checker.check(model.network, properties=properties)
    for checker in [modest_checker, storm_checker]
}

for checker, values in results.items():
    print(f"{checker.description}:")
    for prop_name, prop_value in values.items():
        print(f"  Property {prop_name!r}: {float(prop_value)}")
