from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from FSCUtil import FSCutil
from confidenceMapUtil import FDRutil
import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import time, os

# ********************************
# ******* resolution window ******
# *********************************

class threeeDWindow(QWidget):

	def __init__(self):
		super(threeeDWindow, self).__init__();

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

		#self.symmetry = QLineEdit();
		#self.symmetry.setText('C1');
		#layout.addRow('Symmetry', self.symmetry);


		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());

		self.apix = QLineEdit();
		self.apix.setText('None');
		layout.addRow('Pixel size [A]', self.apix);

		#sampling of azimuth
		self.samplingAzimuth = QLineEdit();
		self.samplingAzimuth.setText('5');
		layout.addRow('# sampling points for azimuth', self.samplingAzimuth);

		#sampling of elevation
		self.samplingElevation = QLineEdit();
		self.samplingElevation.setText('5');
		layout.addRow('# sampling points for elevation', self.samplingElevation);

		#cone opening
		self.coneOpening = QLineEdit();
		self.coneOpening.setText('20');
		layout.addRow('opening of directional cone [˚]', self.coneOpening);

		# add output directory
		hbox_output = QHBoxLayout();
		self.fileLine_output = QLineEdit();
		searchButton_output = self.searchFileButton_output();
		hbox_output.addWidget(self.fileLine_output);
		hbox_output.addWidget(searchButton_output);
		layout.addRow('Save output to ', hbox_output);

		layout.addRow(' ', QHBoxLayout());  # make some space
		layout.addRow(' ', QHBoxLayout());  # make some space
		layout.addRow(' ', QHBoxLayout());  # make some space

		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.FSCBtn();

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);

		formGroupBox = QGroupBox();
		formGroupBox.setLayout(layout);

		#set the main Layout
		heading = QLabel("Directional resolution estimation by FDR-FSC", self);
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

	def searchFileButton_output(self):
		btn = QPushButton('Search directory');
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
		btn.clicked.connect(self.runFSC);

		return btn;

	def showMessageBox(self):

		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Directional resolution estimation finished!");
		msg.setWindowTitle("Results");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	# ---------------------------------------------
	def runFSC(self):

		#show message box before starting
		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Start the job with OK!")
		msg.setInformativeText("GUI will be locked until the job is finished. See terminal printouts for progress ...");
		msg.setWindowTitle("Start job");
		msg.setStandardButtons( QMessageBox.Cancel| QMessageBox.Ok);
		result = msg.exec_();

		if result == QMessageBox.Cancel:
			return;


		start = time.time();

		print('************************************************************');
		print('******* Significance analysis of FSC curves - 3D FSC *******');
		print('************************************************************');

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

		# set working directory and output filename
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine_halfMap1.text());
		os.chdir(path);
		splitFilename = os.path.splitext(os.path.basename(self.fileLine_halfMap1.text()));
		outputFilename_dirRes = splitFilename[0] + "_directionalResolutions.mrc";

		# make the mask
		print("Using a circular mask ...");
		maskData = FSCutil.makeCircularMask(halfMap1Data, (np.min(halfMap1Data.shape) / 2.0) - 4.0);  # circular mask
		maskBFactor = FSCutil.makeCircularMask(halfMap1Data, (
					np.min(halfMap1Data.shape) / 4.0) - 4.0);  # smaller circular mask for B-factor estimation

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


		#********************************************
		#********* get sampling of azimuth **********
		#********************************************

		try:
			samplingAzimuth = int(self.samplingAzimuth.text());
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read sampling of azimuth angles ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		#**********************************************
		#********* get sampling of elevation **********
		#**********************************************

		try:
			samplingElevation = int(self.samplingElevation.text());
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read sampling of elevation angles ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		#****************************************
		#********* get opening of cone **********
		#****************************************

		try:
			coneOpening = int(self.coneOpening.text());
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read sampling of cone opening angle ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		#read the mask
		#mask = mrcfile.open('/Users/mbeckers/Downloads/EMD-3061/other/mask.mrc', mode='r');
		#maskData = mask.data;

		numAsymUnits = 1.0;
		#run the FSC
		phiArray, thetaArray, directionalResolutions, directionalResolutionHeatmap, dirResMap3D = FSCutil.threeDimensionalFSC(halfMap1Data, halfMap2Data,
																				 maskData, apix, 0.143,
																				 numAsymUnits, samplingAzimuth, samplingElevation, coneOpening );

		#plot the directional resolutions
		plt.title("Directional resolution plot")
		plt.imshow(directionalResolutionHeatmap.T, cmap='hot', origin='lower', extent=[-180, 180, 0, 90]);
		cbar  = plt.colorbar(orientation="horizontal");
		cbar.set_label("Resolution [Angstroem]");
		plt.ylabel("Elevation [˚]");
		plt.xlabel("Azimuth [˚]");
		plt.savefig('directionalResolutions.pdf', dpi=400);
		plt.close();

		# write the directional resolution map
		dirResMapMRC = mrcfile.new(outputFilename_dirRes, overwrite=True);
		dirResMap3D = np.float32(dirResMap3D);
		dirResMapMRC.set_data(dirResMap3D);
		dirResMapMRC.voxel_size = apix;
		dirResMapMRC.close();

		output = "Saved local resolutions map to: " + outputFilename_dirRes;
		print(output);

		end = time.time();
		totalRuntime = end - start;

		print("****** Summary ******");
		print("Runtime: %.2f" % totalRuntime);

		self.showMessageBox();

