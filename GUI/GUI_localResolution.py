from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from FSCUtil import FSCutil, localResolutions
from confidenceMapUtil import FDRutil
import mrcfile
import numpy as np
import time, os

# ********************************
# ******* resolution window ******
# *********************************

class ResolutionWindow(QWidget):

	def __init__(self):
		super(ResolutionWindow, self).__init__();
		layout = QFormLayout();

		# ------------ now required input
		layout.addRow(' ', QHBoxLayout()); # make some space
		requiredLabel = QLabel("Required Input", self);
		requiredLabel.setFont(QFont('Arial', 17));
		layout.addRow(requiredLabel, QHBoxLayout());


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

		layout.addRow('', QHBoxLayout()); # make some space


		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());

		self.apix = QLineEdit();
		self.apix.setText('None');
		layout.addRow('Pixel size [A]', self.apix);

		self.stepSize = QLineEdit();
		self.stepSize.setText('5');
		layout.addRow('Step size of sliding window [pixels]', self.stepSize);

		self.w = QLineEdit();
		self.w.setText('20');
		layout.addRow('Size of sliding window [pixels]', self.w);

		self.lowRes = QLineEdit();
		self.lowRes.setText('None');
		layout.addRow('Low resolution limit [A]', self.lowRes);

		# make some space
		layout.addRow('', QHBoxLayout());
		layout.addRow('', QHBoxLayout());
		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.FSCBtn();

		# add mask
		hbox_mask = QHBoxLayout();
		self.fileLine_mask = QLineEdit();
		searchButton_mask = self.searchFileButton_mask();
		hbox_mask.addWidget(self.fileLine_mask);
		hbox_mask.addWidget(searchButton_mask);
		layout.addRow('Mask', hbox_mask);

		# add output directory
		hbox_output = QHBoxLayout();
		self.fileLine_output = QLineEdit();
		searchButton_output = self.searchFileButton_output();
		hbox_output.addWidget(self.fileLine_output);
		hbox_output.addWidget(searchButton_output);
		layout.addRow('Save output to ', hbox_output);

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);


		formGroupBox = QGroupBox();
		formGroupBox.setLayout(layout);

		#set the main Layout
		heading = QLabel("Local resolution estimation by FDR-FSC", self);
		heading.setFont(QFont('Arial', 17));
		heading.setAlignment(Qt.AlignTop);

		mainLayout = QVBoxLayout();
		mainLayout.addWidget(heading);
		mainLayout.addWidget(formGroupBox);
		mainLayout.addLayout(buttonBox);
		self.setLayout(mainLayout);


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

	def searchFileButton_output(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_output);
		return btn;

	def onInputFileButtonClicked_output(self):
		filename = QFileDialog.getExistingDirectory(caption='Set output directory');
		if filename:
			self.fileLine_output.setText(filename);


	def quitButton(self):
		btn = QPushButton('Quit');
		btn.clicked.connect(QCoreApplication.instance().quit);
		btn.resize(btn.minimumSizeHint());

		return btn;

	def FSCBtn(self):

		btn = QPushButton('Run');
		btn.resize(btn.minimumSizeHint());
		btn.clicked.connect(self.runLocalFSC);

		return btn;

	def showMessageBox(self):

		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Local resolutions estimation finished!");
		msg.setWindowTitle("Finished");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	# ---------------------------------------------
	def runLocalFSC(self):

		start = time.time();

		print('***************************************************');
		print('******* Significance analysis of FSC curves *******');
		print('***************************************************');


		#read the half maps
		try:
			half_map1 = mrcfile.open(self.fileLine_halfMap1.text(), mode='r');
			half_map2 = mrcfile.open(self.fileLine_halfMap2.text(), mode='r');
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		halfMap1Data = np.copy(half_map1.data);
		halfMap2Data = np.copy(half_map2.data);
		sizeMap = halfMap1Data.shape;

		# set output filename and working directory
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine_halfMap1.text());
		os.chdir(path);
		splitFilename = os.path.splitext(os.path.basename(self.fileLine_halfMap1.text()));
		outputFilename_LocRes = splitFilename[0] + "_localResolutions.mrc";


		# make the mask
		try:
			mask = mrcfile.open(self.fileLine_mask.text(), mode='r+');
		except:
			mask = None;

		maskData = FSCutil.makeCircularMask(halfMap1Data, (np.min(halfMap1Data.shape) / 2.0) - 4.0);  # circular mask

		if mask is not None:
			print("Using user provided mask ...");
			maskPermutationData = np.copy(mask.data);
		else:
			maskPermutationData = maskData;

		#**************************************
		#********* get pixel size *************
		#**************************************
		apixMap = float(half_map1.voxel_size.x);

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

		#******************************************
		#************* get step Size **************
		#******************************************

		try:
			stepSize = int(self.stepSize.text());
		except:
			print("Invalid input for stepSize. Needs to be a integer >= 0 ...")

		#******************************************
		#************* get step Size **************
		#******************************************

		try:
			stepSize = int(self.stepSize.text());
		except:
			print("Invalid input for stepSize. Needs to be a integer >= 0 ...")

		#********************************************
		#************* get window size **************
		#********************************************

		try:
			windowSize = int(self.w.text());
		except:
			print("Invalid input for windowSize. Needs to be a integer >= 0 ...");

		# ********************************************
		# ************* get window size **************
		# ********************************************

		try:
			lowRes = int(self.lowRes.text());
		except:
			lowRes = None;

		# *******************************************
		# ********* calc local Resolutions **********
		# *******************************************

		FSCcutoff = 0.5;

		localResMap = localResolutions.localResolutions(halfMap1Data, halfMap2Data, windowSize, stepSize, FSCcutoff, apix,
															1,
															maskData, maskPermutationData);

		# set lowest resolution if wished
		if lowRes is not None:
			lowRes = lowRes;
			localResMap[localResMap > lowRes] = lowRes;

		# write the local resolution map
		localResMapMRC = mrcfile.new(outputFilename_LocRes, overwrite=True);
		localResMap = np.float32(localResMap);
		localResMapMRC.set_data(localResMap);
		localResMapMRC.voxel_size = apix;
		localResMapMRC.close();

		output = "Saved local resolutions map to: " + outputFilename_LocRes;
		print(output);

		end = time.time();
		totalRuntime = end - start;

		print("****** Summary ******");
		print("Runtime: %.2f" % totalRuntime);

		self.showMessageBox();