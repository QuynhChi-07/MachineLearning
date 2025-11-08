from PyQt6.QtWidgets import QApplication, QMainWindow

from on_KTGK.UI.MainWindowEx import MainWindowEx

if __name__ == "__main__":
    app = QApplication([])
    mainWin = QMainWindow()
    ui = MainWindowEx()
    ui.setupUi(mainWin)
    mainWin.show()
    app.exec()

