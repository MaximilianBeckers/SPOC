from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import mrcfile
import subprocess
import numpy as np
from confidenceMapUtil import mapUtil

# ********************************
# ***** confidenceMap window *****
# *********************************

class ConfMapWindow(QWidget):

	def __init__(self):
		super(ConfMapWindow, self).__init__();
		layout = QFormLayout();

		# add input file
		hbox = QHBoxLayout();
		self.fileLine = QLineEdit();
		searchButton = self.searchFileButton_inputFilename();
		hbox.addWidget(self.fileLine);
		hbox.addWidget(searchButton);
		layout.addRow('EM Map', hbox);

		# add choice for error criterion
		self.cb = QComboBox();
		self.cb.addItems(['FDR', 'FWER', 'localFDR']);

		layout.addRow('Error criterion:', self.cb);

		self.apix = QLineEdit();
		self.apix.setText('1');
		layout.addRow('Pixel size [A]', self.apix);

		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		layout.addRow('Optional Input:', QHBoxLayout());

		# add local resolution
		hbox = QHBoxLayout();
		self.fileLine_localRes = QLineEdit();
		searchButton_localRes = self.searchFileButton_localResFilename();
		hbox.addWidget(self.fileLine_localRes);
		hbox.addWidget(searchButton_localRes);
		layout.addRow('Local Resolution Map', hbox);

		# add box size for background noise estimation
		self.boxSize = QLineEdit();
		self.boxSize.setText('0');
		layout.addRow('Box size:', self.boxSize);

		# add box coordinates
		coordBox = QHBoxLayout();
		self.xCoord = QLineEdit();
		self.yCoord = QLineEdit();
		self.zCoord = QLineEdit();
		self.xCoord.setText('0'), self.yCoord.setText('0'), self.zCoord.setText('0');
		coordBox.addWidget(self.xCoord);
		coordBox.addWidget(self.yCoord);
		coordBox.addWidget(self.zCoord);
		layout.addRow('Box coordinates [x][y][z]:', coordBox);

		# make some space
		layout.addRow('',QHBoxLayout());
		layout.addRow('',QHBoxLayout());

		# some buttons
		qtBtn = self.quitButton();
		runBtn = QPushButton('Run');
		runBtn.resize(runBtn.minimumSizeHint());

		checkBtn = self.checkNoiseEstimationBtn();

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);
		buttonBox.addWidget(checkBtn);

		layout.addRow(' ', buttonBox);
		self.setLayout(layout);

	def searchFileButton_inputFilename(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_inputFilename);
		return btn;

	def onInputFileButtonClicked_inputFilename(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine.setText(filename[0]);

	def searchFileButton_localResFilename(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_localResFilename);
		return btn;

	def onInputFileButtonClicked_localResFilename(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');
		if filename:
			self.fileLine_localRes.setText(filename[0]);

	def quitButton(self):
		btn = QPushButton('Quit');
		btn.clicked.connect(QCoreApplication.instance().quit);
		btn.resize(btn.minimumSizeHint());

		return btn;

# --------------------------------------------------------
	def checkNoiseEstimationBtn(self):
		btn = QPushButton('Check Noise Estim.');
		btn.clicked.connect(self.checkNoiseEstimation);
		btn.resize(btn.minimumSizeHint());

		return btn;

	def checkNoiseEstimation(self):

		print('Check background noise estimation ...');
		try:
			map = mrcfile.open(self.fileLine.text(), mode='r+');
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		mapData = np.copy(map.data);
		sizeMap = mapData.shape;

		#make the circular mask
		sphere_radius = (np.max(sizeMap) // 2);
		circularMaskData = mapUtil.makeCircularMask( np.copy(mapData), sphere_radius);

		windowSize = int(self.boxSize.text());

		boxCoord = np.array([int(self.xCoord.text()), int(self.yCoord.text()), int(self.zCoord.text())]) ;
		print(boxCoord);

		#generate the diagnostic image
		mapUtil.makeDiagnosticPlot(mapData, windowSize, 0, False, boxCoord, circularMaskData);

		subprocess.call(["open", "diag_image.pdf"]);

