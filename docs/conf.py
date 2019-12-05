import os
import sys

sys.path.insert(0, os.path.join(os.path.abspath('.'), '..', '..'))


project = 'Momba'
copyright = '2019, Maximilian Köhl'
author = 'Maximilian Köhl'

release = '0.0.1dev'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]


templates_path = []  # type: ignore


html_theme = 'sphinx_rtd_theme'

html_static_path = []  # type: ignore
