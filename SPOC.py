import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from GUI import GUI_localFiltering, GUI_resolutions, GUI_confidenceMap, GUI_localResolution

class Window(QWidget):

	def __init__(self):

		super(Window, self).__init__();



		#set first line of GUI
		self.nameLabel = QLabel("SPOC", self);

		self.captionLayout = QHBoxLayout();
		self.captionLayout.addWidget(self.nameLabel);




		#set second line of GUI
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

		self.mainLayout = QHBoxLayout();
		self.mainLayout.addWidget(self.leftlist);
		self.mainLayout.addWidget(self.Stack);


		#set third line of GUI
		logoEMBL = QLabel(self)
		pixmap = QPixmap('GUI/images/EMBL_logo.png')
		pixmap_scaled = pixmap.scaledToWidth(200)
		logoEMBL.setPixmap(pixmap_scaled);

		logoFZ = QLabel(self)
		pixmap = QPixmap('GUI/images/fz_logo.png')
		pixmap_scaled = pixmap.scaledToWidth(200)
		logoFZ.setPixmap(pixmap_scaled);


		self.bottomLayout = QHBoxLayout();
		self.bottomLayout.addWidget(logoEMBL);
		self.bottomLayout.addStretch(1);
		self.bottomLayout.addWidget(logoFZ);



		#set overall layout
		self.overallLayout = QVBoxLayout();
		self.overallLayout.addLayout(self.captionLayout);
		self.overallLayout.addLayout(self.mainLayout);
		self.overallLayout.addLayout(self.bottomLayout);

		self.setLayout(self.overallLayout);
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
