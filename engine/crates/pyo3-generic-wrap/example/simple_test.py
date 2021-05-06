import pyo3_generic_wrap_example

with_a = pyo3_generic_wrap_example.MyClass.new_a()
print(with_a.say_hello("World"))

with_b = pyo3_generic_wrap_example.MyClass.new_b()
print(with_b.say_hello("Hello"))
