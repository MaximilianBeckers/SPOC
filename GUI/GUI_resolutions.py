import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from FSCUtil import FSCutil
import mrcfile
import numpy as np
import time
import os.path

# ********************************
# ******* resolution window ******
# *********************************

class ResolutionWindow(QWidget):

	def __init__(self):
		super(ResolutionWindow, self).__init__();
		layout = QFormLayout();

		# add input file
		hbox_half1 = QHBoxLayout();
		self.fileLine_halfMap1 = QLineEdit();
		searchButton_halfMap1 = self.searchFileButton_halfMap1();
		hbox_half1.addWidget(self.fileLine_halfMap1);
		hbox_half1.addWidget(searchButton_halfMap1);
		layout.addRow('Half Map 1', hbox_half1);


		# add second half map
		hbox_half2 = QHBoxLayout();
		self.fileLine_halfMap2 = QLineEdit();
		searchButton_halfMap2 = self.searchFileButton_halfMap2();
		hbox_half2.addWidget(self.fileLine_halfMap2);
		hbox_half2.addWidget(searchButton_halfMap2);
		layout.addRow('Half Map 2', hbox_half2);

		self.apix = QLineEdit();
		layout.addRow('Pixel size [A]', self.apix);

		#optional FSC input
		layout.addRow('',QHBoxLayout());
		layout.addRow('',QHBoxLayout());

		layout.addRow(' ', QHBoxLayout()); # make some space
		layout.addRow('Optional Input:', QHBoxLayout());

		self.numAsUnit = QLineEdit();
		self.numAsUnit.setText('1');
		layout.addRow('# asym. units', self.numAsUnit);

		# add second half map
		hbox_mask = QHBoxLayout();
		self.fileLine_mask = QLineEdit();
		searchButton_mask = self.searchFileButton_mask();
		hbox_mask.addWidget(self.fileLine_mask);
		hbox_mask.addWidget(searchButton_mask);
		layout.addRow('Mask', hbox_mask);

		#local FSC input
		layout.addRow('',QHBoxLayout());
		layout.addRow('',QHBoxLayout());

		layout.addRow(' ', QHBoxLayout()); # make some space
		layout.addRow('Local FSC Input:', QHBoxLayout());

		self.boxSize = QLineEdit();
		self.boxSize.setText('20');
		layout.addRow('Box Size', self.boxSize);

		self.stepSize = QLineEdit();
		self.stepSize.setText('5');
		layout.addRow('Step Size', self.stepSize);

		self.FSCcutOff = QLineEdit();
		self.FSCcutOff.setText('0.5');
		layout.addRow('FSC cutoff', self.FSCcutOff);

		# make some space
		layout.addRow('', QHBoxLayout());
		layout.addRow('', QHBoxLayout());

		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.FSCBtn();
		localResBtn = self.localResBtn();

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);
		buttonBox.addWidget(localResBtn);

		layout.addRow(' ', buttonBox);
		self.setLayout(layout);

	def searchFileButton_halfMap1(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_halfMap1);
		return btn;

	def onInputFileButtonClicked_halfMap1(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine_halfMap1.setText(filename[0]);

	def searchFileButton_halfMap2(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_halfMap2);
		return btn;

	def onInputFileButtonClicked_halfMap2(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');
		if filename:
			self.fileLine_halfMap2.setText(filename[0]);

	def searchFileButton_mask(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_mask);
		return btn;

	def onInputFileButtonClicked_mask(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');
		if filename:
			self.fileLine_mask.setText(filename[0]);


	def quitButton(self):
		btn = QPushButton('Quit');
		btn.clicked.connect(QCoreApplication.instance().quit);
		btn.resize(btn.minimumSizeHint());

		return btn;

	def FSCBtn(self):

		btn = QPushButton('Run FSC');
		btn.resize(btn.minimumSizeHint());
		btn.clicked.connect(self.runFSC);

		return btn;

	# ---------------------------------------------
	def runFSC(self):

		#read the half maps
		try:
			half_map1 = mrcfile.open(self.fileLine_halfMap1.text(), mode='r+');
			half_map2 = mrcfile.open(self.fileLine_halfMap2.text(), mode='r+');
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		mapData_half1 = np.copy(half_map1.data);
		mapData_half2 = np.copy(half_map2.data);
		sizeMap = mapData_half1.shape;

		#get the mask
		try:
			mask = mrcfile.open(self.fileLine_mask.text(), mode='r+');
			maskData = np.copy(mask.data);

		except:
			#maskData = np.ones(mapData_half1.shape);
			maskData = FSCutil.makeCircularMask(mapData_half1, (np.min(sizeMap)/2.0)-4.0);



		#get pixel size
		apix = float(self.apix.text());
		numAsymUnits = float(self.numAsUnit.text());

		#res, FSC, percentCutoffs, threeSigma, threeSigmaCorr, resolution, _, _ = FSCutil.FSC(mapData_half1, mapData_half2, maskData, apix, 0.143, numAsymUnits, False, True, None);

		#write the FSC
		#FSCutil.writeFSC(res, FSC, percentCutoffs, threeSigmaCorr);

		FSCutil.simulatedVolumes(maskData, numAsymUnits);
		#FSCutil.comparison_sizeVolumes(mapData_half1, mapData_half2, apix);
		#FSCutil.comparisonSignalWithNoise(maskData, numAsymUnits);
		#FSCutil.effSampleSize(maskData, 1);

	def localResBtn(self):

		btn = QPushButton('Run local FSC');
		btn.resize(btn.minimumSizeHint());
		btn.clicked.connect(self.runLocalFSC);

		return btn;

	# -------------------------------------------------
	def runLocalFSC(self):

		#print(self.FSCcutOff.text());
		start = time.time();

		#read the half maps
		try:
			half_map1 = mrcfile.open(self.fileLine_halfMap1.text(), mode='r+');
			half_map2 = mrcfile.open(self.fileLine_halfMap2.text(), mode='r+');
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		splitFilename = os.path.splitext(os.path.basename(self.fileLine_halfMap1.text()));
		outputFilename = splitFilename[0] + "_localResolutions.mrc";

		mapData_half1 = np.copy(half_map1.data);
		mapData_half2 = np.copy(half_map2.data);
		sizeMap = mapData_half1.shape;

		#get the mask
		try:
			mask = mrcfile.open(self.fileLine_mask.text(), mode='r+');
			maskData = np.copy(mask.data);
		except:
			print("Using a circular mask ...");
			maskData = FSCutil.makeCircularMask(mapData_half1, (np.min(sizeMap)/2.0)-4.0);


		#get some numerical input
		apix = float(self.apix.text());
		numAsymUnits = float(self.numAsUnit.text());
		boxSize = int(self.boxSize.text());
		stepSize = int(self.stepSize.text());
		cutoff = float(self.FSCcutOff.text());

		localResMap = FSCutil.localResolutions(mapData_half1, mapData_half2, boxSize, stepSize, cutoff, apix, numAsymUnits, maskData);

		#write the local resolution map
		localResMapMRC = mrcfile.new(outputFilename, overwrite=True);
		localResMap = np.float32(localResMap);
		localResMapMRC.set_data(localResMap);
		localResMapMRC.voxel_size = apix;
		localResMapMRC.close();

		end = time.time();
		totalRuntime = end - start;

		print("Runtime: %.2f" %totalRuntime);
