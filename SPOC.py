import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from GUI import GUI_localFiltering, GUI_resolutions, GUI_confidenceMap, GUI_localResolution

class Window(QWidget):

	def __init__(self):

		super(Window, self).__init__();
		self.leftlist = QListWidget();
		self.leftlist.insertItem(0, 'Global resolution estimation by FDR-FSC');
		self.leftlist.insertItem(1, 'Local resolution estimation by FDR-FSC');
		self.leftlist.insertItem(2, 'Local resolution filtering');
		self.leftlist.insertItem(3, 'Confidence maps');

		self.stack1 = GUI_resolutions.ResolutionWindow();
		self.stack2 = GUI_localResolution.ResolutionWindow();
		self.stack3 = GUI_localFiltering.LocalFilteringWindow();
		self.stack4 = GUI_confidenceMap.ConfMapWindow();

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
		self.setWindowTitle('SPOC');
		self.show();
	
	def display(self,i):
		self.Stack.setCurrentIndex(i);

def main():
	
	app = QApplication(sys.argv);
	GUI = Window();
	sys.exit(app.exec_());

if __name__ == '__main__':
	main();
