import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from GUI import GUI_uniformSharpening, GUI_mapDoseWeighting, GUI_resolutions, GUI_confidenceMap

class Window(QWidget):
	
	def __init__(self):

		super(Window, self).__init__();
		self.leftlist = QListWidget();
		self.leftlist.insertItem(0, 'Resolution measures');
		self.leftlist.insertItem(1, 'Uniform Sharpening and Filtering');
		self.leftlist.insertItem(2, 'Local Filtering');
		self.leftlist.insertItem(3, 'LocScale');
		self.leftlist.insertItem(4, 'ConfidenceMap');
		self.leftlist.insertItem(5, 'Map Dose-Weighting');

		self.stack1 = GUI_resolutions.ResolutionWindow();
		self.stack2 = GUI_uniformSharpening.SharpeningWindow();
		self.stack3 = GUI_confidenceMap.ConfMapWindow();
		self.stack4 = GUI_mapDoseWeighting.MapDWWindow();

		self.Stack = QStackedWidget(self);
		self.Stack.addWidget(self.stack1);
		self.Stack.addWidget(self.stack2);
		self.Stack.addWidget(self.stack3);
		self.Stack.addWidget(self.stack4);
		
		hbox = QHBoxLayout(self);
		hbox.addWidget(self.leftlist);
		hbox.addWidget(self.Stack);
		
		self.setLayout(hbox);
		self.leftlist.currentRowChanged.connect(self.display);
		self.setWindowTitle('PoPro');
		self.show();
	
	def display(self,i):
		self.Stack.setCurrentIndex(i);

def main():
	
	app = QApplication(sys.argv);
	GUI = Window();
	sys.exit(app.exec_());

if __name__ == '__main__':
	main();
