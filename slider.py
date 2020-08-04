import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                             QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider)
min_value=0
class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        grid = QGridLayout()
        grid.addWidget(self.createExampleGroup(), 0, 0)
        
        self.setLayout(grid)

        self.setWindowTitle("Slider")
        self.resize(400, 100)

    def createExampleGroup(self):
        groupBox = QGroupBox("Set Minimum Distance")


        slider = QSlider(Qt.Horizontal)
        slider.setMaximum(20)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(1)
        slider.setSingleStep(1)
        slider.valueChanged.connect(self.scaletext)

        vbox = QVBoxLayout()
        #vbox.addWidget(radio1)
        vbox.addWidget(slider)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
    def scaletext(self, value):
        global min_value
        min_value=value
        f = open("minValue.txt", "w")
        f.write(str(min_value))
        
        
        
        print(min_value)
def return_value():
    global min_value
    return(min_value)

app = QApplication(sys.argv)
clock = Window()
clock.show()
sys.exit(app.exec_())
