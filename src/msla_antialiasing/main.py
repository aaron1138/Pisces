import sys
import os
from PySide6.QtWidgets import QApplication

# Add the parent directory ('src') to the Python path
# This allows the script to be run from inside the package directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from msla_antialiasing.ui.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
