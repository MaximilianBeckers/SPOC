from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from FSCUtil import FSCutil
from confidenceMapUtil import FDRutil
import mrcfile
import numpy as np
import time, os

# ********************************
# ******* sharpening window ******
# ********************************

class SharpeningWindow(QWidget):

	def __init__(self):
		super(SharpeningWindow, self).__init__();

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


		self.resolution = QLineEdit();
		self.resolution.setText('2');
		layout.addRow('Resolution [A]', self.resolution);

		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());

		self.apix = QLineEdit();
		self.apix.setText('None');
		layout.addRow('Pixel size [A]', self.apix);

		self.bFactor = QLineEdit();
		self.bFactor.setText('None');
		layout.addRow('B-factor', self.bFactor);

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
		heading = QLabel("Global sharpening", self);
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
		btn.clicked.connect(self.runFSC);

		return btn;


	def showMessageBox(self, resolution, bFactor):

		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Map filtered at a resolution of: {:.2f} A. \n \nEstimated B-factor: {:.2f}".format(resolution, bFactor));
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

		print('***************************************************');
		print('**********  Sharpening of cryo-EM maps  ***********');
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

		# set working directory and output filename
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine_halfMap1.text());
		os.chdir(path);
		splitFilename = os.path.splitext(os.path.basename(self.fileLine_halfMap1.text()));
		outputFilename_PostProcessed =  "postProcessed.mrc";
		outputFilename_PostProcessed_half1 = "postProcessed_half1.mrc";
		outputFilename_PostProcessed_half2 = "postProcessed_half2.mrc";

		# make the mask
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

		#**********************************************
		#***************** get bfactor ****************
		#**********************************************
		try:
			bFactorInput = float(self.bFactor.text());
		except:
			bFactorInput = None;

		#**********************************************
		#***************** get bfactor ****************
		#**********************************************
		try:
			resolution = float(self.resolution.text());
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("No resolution specified ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;


		if (resolution > 8.0) and (bFactorInput is None):
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Automated B-factor estimation is unstable for low-resolution maps. Please specify a B-factor!");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;


		if bFactorInput is not None:
			bFactor = bFactorInput;
			bFactor_half1 = bFactorInput;
			bFactor_half2 = bFactorInput;
			print('Using a user-specified B-factor of {:.2f} for map sharpening'.format(-bFactor));
		else:
			# estimate b-factor and sharpen the map
			bFactor = FSCutil.estimateBfactor(0.5 * (halfMap1Data + halfMap2Data), resolution, apix, maskBFactor);
			print('Using a B-factor of {:.2f} for map sharpening.'.format(-bFactor));

			#bFactor_half1 = FSCutil.estimateBfactor(halfMap1Data, resolution, apix, maskBFactor);
			#bFactor_half2 = FSCutil.estimateBfactor(halfMap2Data, resolution, apix, maskBFactor);

			#print("B-factor of halfmap 1: {:.2f}".format(bFactor_half1));
			#print("B-factor of halfmap 2: {:.2f}".format(bFactor_half2));

		processedMap = FDRutil.sharpenMap(0.5 * (halfMap1Data + halfMap2Data), -bFactor, apix, resolution);
		#processed_halfMap1 = FDRutil.sharpenMap(halfMap1Data, -bFactor_half1, apix, resolution);
		#processed_halfMap2 = FDRutil.sharpenMap(halfMap2Data, -bFactor_half2, apix, resolution);

		# write the post-processed map
		postProcMRC = mrcfile.new(outputFilename_PostProcessed, overwrite=True);
		postProc = np.float32(processedMap);
		postProcMRC.set_data(postProc);
		postProcMRC.voxel_size = apix;
		postProcMRC.close();

		"""
		# write the post-processed halfmaps
		postProcMRC = mrcfile.new(outputFilename_PostProcessed_half1, overwrite=True);
		postProc = np.float32(processed_halfMap1);
		postProcMRC.set_data(postProc);
		postProcMRC.voxel_size = apix;
		postProcMRC.close();

		postProcMRC = mrcfile.new(outputFilename_PostProcessed_half2, overwrite=True);
		postProc = np.float32(processed_halfMap2);
		postProcMRC.set_data(postProc);
		postProcMRC.voxel_size = apix;
		postProcMRC.close();
		"""

		output = "Saved sharpened and filtered map to: " + outputFilename_PostProcessed;
		print(output);

		end = time.time();
		totalRuntime = end - start;

		print("****** Summary ******");
		print("Runtime: %.2f" % totalRuntime);

		self.showMessageBox(resolution, bFactor);

