# File: App.py
import sys
from PyQt6 import QtWidgets
from MainWindowEx import MainWindowEx

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindowEx()
    main_window.show()
    sys.exit(app.exec())