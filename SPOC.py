import sys, os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from GUI import GUI_localFiltering, GUI_resolutions, GUI_confidenceMap, GUI_localResolution, GUI_SMLM

class Window(QWidget):

	def __init__(self):

		super(Window, self).__init__();


		#set first line of GUI
		self.nameLabel = QLabel("Statistical Processing of cryo-EM maps", self);
		self.nameLabel.setFont(QFont('Helvetica', 25));

		self.captionLayout = QHBoxLayout();
		self.captionLayout.addWidget(self.nameLabel);
		self.captionLayout.addStretch(1);

		#set second line of GUI
		#---------------------------------------------
		self.EMlabel = QLabel("cryo-EM", self);
		self.EMlabel.setFont(QFont('Arial', 15));

		self.leftlistEM = QListWidget();
		self.leftlistEM.insertItem(0, 'Global resolution estimation by FDR-FSC');
		self.leftlistEM.insertItem(1, 'Local resolution estimation by FDR-FSC');
		self.leftlistEM.insertItem(2, 'Local resolution filtering');
		self.leftlistEM.insertItem(3, 'Confidence maps');

		self.LMlabel = QLabel("Single Molecule Localization Microscopy", self);
		self.LMlabel.setFont(QFont('Arial', 15));

		self.leftlistLM = QListWidget();
		self.leftlistLM.insertItem(0, 'Global resolution estimation by FDR-FSC');
		self.leftlistLM.insertItem(1, 'Local resolution estimation by FDR-FSC');

		self.leftLayout = QVBoxLayout();
		self.leftLayout.addWidget(self.EMlabel);
		self.leftLayout.addWidget(self.leftlistEM);
		self.leftLayout.addWidget(self.LMlabel);
		self.leftLayout.addWidget(self.leftlistLM);


		self.stack1 = GUI_resolutions.ResolutionWindow();
		self.stack2 = GUI_localResolution.ResolutionWindow();
		self.stack3 = GUI_localFiltering.LocalFilteringWindow();
		self.stack4 = GUI_confidenceMap.ConfMapWindow();
		self.stack5 = GUI_SMLM.SMLMResolutionWindow();


		self.Stack = QStackedWidget(self);
		self.Stack.addWidget(self.stack1);
		self.Stack.addWidget(self.stack2);
		self.Stack.addWidget(self.stack3);
		self.Stack.addWidget(self.stack4);
		self.Stack.addWidget(self.stack5);

		self.mainLayout = QHBoxLayout();
		self.mainLayout.addLayout(self.leftLayout);
		self.mainLayout.addWidget(self.Stack);



		if getattr(sys, 'frozen', False):
			# we are running in a bundle
			path = sys.executable;
			path = os.path.dirname(path);
		else:
			# we are running in a normal Python environment
			path = os.path.dirname(os.path.abspath(__file__)) + "/GUI";

		#set third line of GUI
		logoEMBL = QLabel(self);
		filename_logoEMBL = os.path.normcase(path + "/EMBL_logo.png");
		pixmap = QPixmap(filename_logoEMBL);
		pixmap_scaled = pixmap.scaledToWidth(200);
		logoEMBL.setPixmap(pixmap_scaled);

		logoFZ = QLabel(self)
		filename_logoFZ = os.path.normcase(path + "/fz_logo.png");
		pixmap = QPixmap(filename_logoFZ);
		pixmap_scaled = pixmap.scaledToWidth(200);
		logoFZ.setPixmap(pixmap_scaled);


		self.bottomLayout = QHBoxLayout();
		self.bottomLayout.addWidget(logoEMBL);
		self.bottomLayout.addStretch(1);
		self.bottomLayout.addWidget(QLabel(" Maximilian Beckers\n maximilian.beckers@embl.de", self));
		self.bottomLayout.addStretch(1);
		self.bottomLayout.addWidget(logoFZ);

		#set overall layout
		self.overallLayout = QVBoxLayout();
		self.overallLayout.addLayout(self.captionLayout);
		self.overallLayout.addSpacing(20);
		self.overallLayout.addLayout(self.mainLayout);
		self.overallLayout.addLayout(self.bottomLayout);

		self.setLayout(self.overallLayout);
		self.leftlistEM.itemClicked.connect(self.displayEM);
		self.leftlistLM.itemClicked.connect(self.displayLM);
		self.setWindowTitle('SPOC');
		self.show();

	def displayEM(self):
		i = self.leftlistEM.currentRow();

		self.Stack.setCurrentIndex(i);

	def displayLM(self):

		i = self.leftlistLM.currentRow();
		self.Stack.setCurrentIndex(i+4);

def main():

	app = QApplication(sys.argv);
	GUI = Window();
	sys.exit(app.exec_());

if __name__ == '__main__':
	main();
