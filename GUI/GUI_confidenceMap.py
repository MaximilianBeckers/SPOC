from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import mrcfile
import subprocess
import numpy as np
import time, os
from confidenceMapUtil import mapUtil, FDRutil, confidenceMapMain

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

		# add box size for background noise estimation
		self.boxSize = QLineEdit();
		self.boxSize.setText('50');
		layout.addRow('Box size:', self.boxSize);

		# add choice for error criterion
		self.cb = QComboBox();
		self.cb.addItems(['FDR Benj.-Yekut.', 'FDR Benj.-Hochb.', 'FWER Bonf.-Holm', 'FWER Hochberg']);
		layout.addRow('Error criterion:', self.cb);


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

		self.apix = QLineEdit();
		self.apix.setText('None');
		layout.addRow('Pixel size [A]', self.apix);

		self.ECDF = QCheckBox(self);
		layout.addRow('Use non-parametric EDCF estimation?', self.ECDF);

		# add box coordinates
		coordBox = QHBoxLayout();
		self.xCoord = QLineEdit();
		self.yCoord = QLineEdit();
		self.zCoord = QLineEdit();
		self.xCoord.setText('None'), self.yCoord.setText('None'), self.zCoord.setText('None');
		coordBox.addWidget(self.xCoord);
		coordBox.addWidget(self.yCoord);
		coordBox.addWidget(self.zCoord);
		layout.addRow('Box coordinates [x][y][z]:', coordBox);

		# make some space
		layout.addRow('',QHBoxLayout());
		layout.addRow('',QHBoxLayout());

		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.runButton();

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

	def runButton(self):
		btn = QPushButton('Run');
		btn.clicked.connect(self.run);
		btn.resize(btn.minimumSizeHint());

		return btn;

	def showMessageBox(self):
		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Confidence map calculation finished!");
		msg.setWindowTitle("Finished");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	#--------------------------------------------------------
	def checkNoiseEstimationBtn(self):
		btn = QPushButton('Check Noise Estim.');
		btn.clicked.connect(self.checkNoiseEstimation);
		self.dialogs = list();

		btn.resize(btn.minimumSizeHint());

		return btn;

	def checkNoiseEstimation(self):

		print('Check background noise estimation ...');
		try:
			map = mrcfile.open(self.fileLine.text(), mode='r');
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

		try:
			windowSize = int(self.boxSize.text());
		except:
			print("Window size needs to be a positive integer ...")
			return;

		try:
			boxCoord = [int(self.xCoord.text()), int(self.yCoord.text()), int(self.zCoord.text())] ;
		except:
			boxCoord = 0;

		#generate the diagnostic image
		pp = mapUtil.makeDiagnosticPlot(mapData, windowSize, False, boxCoord);
		pp.savefig('diag_image.png');

		#now show the diagnostic image in new window
		dialog = NoiseWindow(self)
		self.dialogs.append(dialog);
		dialog.show();

		#subprocess.call(["open", "diag_image.pdf"]);


	#-----------------------------------------------------------
	#------------------ run confidence map code ----------------
	#-----------------------------------------------------------
	def run(self):

		start = time.time();

		print('************************************************');
		print('******* Significance analysis of EM-Maps *******');
		print('************************************************');


		# read the EM map
		try:
			em_map = mrcfile.open(self.fileLine.text(), mode='r');
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		mapData = np.copy(em_map.data);

		# read the localResMap map
		try:
			localResMap = mrcfile.open(self.fileLine_localRes.text(), mode='r');
			locResMapData = np.copy(localResMap.data);
		except:
			locResMapData = None;

		# **************************************
		# ********* get pixel size *************
		# **************************************
		apixMap = float(em_map.voxel_size.x);

		try:
			apix = float(self.apix.text());
		except:
			apix = None;

		if apix is not None:
			print('Pixel size set to {:.3f} Angstroem. (Pixel size encoded in map: {:.3f})'.format(apix, apixMap));
		else:
			print(
				'Pixel size was read as {:.3f} Angstroem. If this is incorrect, please specify with -p pixelSize'.format(
					apixMap));
			apix = apixMap;

		#****************************************
		#************ set the method ************
		#****************************************

		if self.cb.currentText() == 'FDR Benj.-Yekut.':
			method = 'BY';
		elif self.cb.currentText() == 'FDR Benj.-Hochb.':
			method = 'BH';
		elif self.cb.currentText() == 'FWER Bonf.-Holm':
			method = 'Holm';
		elif self.cb.currentText() == 'FWER Hochberg':
			method = 'Hochberg';


		#****************************************
		#************ set the noiseBox **********
		#****************************************
		try:
			boxCoord = [int(self.xCoord.text()), int(self.yCoord.text()), int(self.zCoord.text())] ;
		except:
			boxCoord = 0;

		# ******************************************
		# ************ set the windowSize **********
		# ******************************************
		try:
			windowSize = int(self.boxSize.text());
		except:
			print("Window size needs to be a positive integer ...");
			return;


		#set filename for output
		splitFilename = os.path.splitext(os.path.basename(self.fileLine.text()));

		#*******************************************
		# ******** run the actual analysis *********
		#*******************************************
		confidenceMap, locFiltMap, locScaleMap, mean, var = confidenceMapMain.calculateConfidenceMap(mapData, apix,
																									 boxCoord,
																									 "rightSided",
																									 self.ECDF.isChecked(),
																									 None,
																									 method,
																									 windowSize,
																									 locResMapData,
																									 None,
																									 None,
																									 None,
																									 None,
																									 None,
																									 None,
																									 False);

		# write the confidence Maps
		confidenceMapMRC = mrcfile.new(splitFilename[0] + '_confidenceMap.mrc', overwrite=True);
		confidenceMap = np.float32(confidenceMap);
		confidenceMapMRC.set_data(confidenceMap);
		confidenceMapMRC.voxel_size = apix;
		confidenceMapMRC.close();

		end = time.time();
		totalRuntime = end - start;

		print("****** Summary ******");
		print("Runtime: %.2f" % totalRuntime);

		self.showMessageBox();

#-------------------------------------------------------
#------ window for background noise estimation ---------
#-------------------------------------------------------

class NoiseWindow(QWidget):
	def __init__(self, parent=None):
		super(NoiseWindow, self).__init__();

		self.label = QLabel(self)
		self.label.setPixmap(QPixmap("diag_image.png"));

		vbox = QVBoxLayout()
		vbox.addWidget(self.label);
		self.setLayout(vbox)

		self.setWindowTitle("Check background noise estimation")
		self.show()