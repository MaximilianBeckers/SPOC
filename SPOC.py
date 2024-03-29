import sys, os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from GUI import GUI_localFiltering, GUI_resolutions, GUI_sharpening, GUI_confidenceMap, GUI_localResolution, GUI_SMLM, GUI_SMLM_localResolution, GUI_3DFSC


class Window(QWidget):

	def __init__(self):

		#set path for images
		if getattr(sys, 'frozen', False):
			# we are running in a bundle
			path = sys.executable;
			path = os.path.dirname(path);
		else:
			# we are running in a normal Python environment
			path = os.path.dirname(os.path.abspath(__file__)) + "/GUI";

		super(Window, self).__init__();


		""""#set backgorund
		oImage = QImage(path + "/TMV_in_acid.jpg")
		oImage = oImage.scaled(QSize(1500, 1500))  # resize Image to widgets size
		palette = QPalette()
		palette.setBrush(self.backgroundRole(), QBrush(oImage))  # 10 = Windowrole
		self.setPalette(palette)
		self.setWindowOpacity(0.93)
		self.setStyleSheet("color: white;")
		"""

		#set first line of GUI
		logo = QLabel(self);
		filename_logo = os.path.normcase(path + "/Logo.png");
		pixmap = QPixmap(filename_logo);
		pixmap_scaled = pixmap.scaledToWidth(320);
		logo.setPixmap(pixmap_scaled);

		self.captionLayout = QHBoxLayout();
		#self.captionLayout.addStretch(1);
		self.captionLayout.addWidget(logo);
		self.captionLayout.addStretch(1);

		#set second line of GUI
		#---------------------------------------------
		self.EMlabel = QLabel("cryo-EM", self);
		self.EMlabel.setFont(QFont('Arial', 17));

		self.leftlistEM = QListWidget();
		self.leftlistEM.insertItem(0, 'Global resolution estimation by FDR-FSC');
		self.leftlistEM.insertItem(1, 'Local resolution estimation by FDR-FSC');
		self.leftlistEM.insertItem(2, '3D FSC - Directional resolutions by FDR-FSC');
		self.leftlistEM.insertItem(3, 'Sharpening');
		self.leftlistEM.insertItem(4, 'Local resolution filtering');
		self.leftlistEM.insertItem(5, 'Confidence maps');

		#self.LMlabel = QLabel("Single Molecule Localization Microscopy", self);
		#self.LMlabel.setFont(QFont('Arial', 17));

		self.leftlistLM = QListWidget();
		self.leftlistLM.insertItem(0, 'Global resolution estimation by FDR-FSC');
		self.leftlistLM.insertItem(1, 'Local resolution estimation by FDR-FSC');

		self.leftLayout = QVBoxLayout();
		self.leftLayout.addWidget(self.EMlabel);
		self.leftLayout.addWidget(self.leftlistEM);
		#self.leftLayout.addWidget(self.LMlabel);
		#self.leftLayout.addWidget(self.leftlistLM);


		self.stack1 = GUI_resolutions.ResolutionWindow();
		self.stack2 = GUI_localResolution.ResolutionWindow();
		self.stack3 = GUI_3DFSC.threeeDWindow();
		self.stack4 = GUI_sharpening.SharpeningWindow();
		self.stack5 = GUI_localFiltering.LocalFilteringWindow();
		self.stack6 = GUI_confidenceMap.ConfMapWindow();
		self.stack7 = GUI_SMLM.SMLMResolutionWindow();
		self.stack8 = GUI_SMLM_localResolution.SMLMLocalResolutionWindow();


		self.Stack = QStackedWidget(self);
		self.Stack.addWidget(self.stack1);
		self.Stack.addWidget(self.stack2);
		self.Stack.addWidget(self.stack3);
		self.Stack.addWidget(self.stack4);
		self.Stack.addWidget(self.stack5);
		self.Stack.addWidget(self.stack6);
		self.Stack.addWidget(self.stack7);
		self.Stack.addWidget(self.stack8);

		self.mainLayout = QHBoxLayout();
		self.mainLayout.addLayout(self.leftLayout);
		self.mainLayout.addWidget(self.Stack);


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
		self.Stack.setCurrentIndex(i+6);

def main():

	app = QApplication(sys.argv);
	GUI = Window();
	sys.exit(app.exec_());

if __name__ == '__main__':
	main();
