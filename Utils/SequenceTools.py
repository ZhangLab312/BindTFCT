import os


def cell_label():
    rooT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    cells = os.listdir(rooT + '\\' + 'Dataset')
    cells.remove('script.py')
    cells.remove('__pycache__')

    for cell in cells:
        print(cell)


cell_label()
