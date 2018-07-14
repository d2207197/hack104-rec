from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader('hack104_rec', 'template'),
    autoescape=select_autoescape(['html', 'xml', 'j2'])
)
