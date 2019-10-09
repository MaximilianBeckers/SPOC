from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from FSCUtil import FSCutil
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

		self.symmetry = QLineEdit();
		self.symmetry.setText('C1');
		layout.addRow('Symmetry', self.symmetry);

		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());

		self.apix = QLineEdit();
		self.apix.setText('None');
		layout.addRow('Pixel size [A]', self.apix);

		self.numAsUnit = QLineEdit();
		self.numAsUnit.setText('None');
		layout.addRow('# asym. units', self.numAsUnit);

		self.bFactor = QLineEdit();
		self.bFactor.setText('None');
		layout.addRow('B-factor', self.bFactor);

		# make some space
		layout.addRow('', QHBoxLayout());
		layout.addRow('', QHBoxLayout());

		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.FSCBtn();

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);

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


	def showMessageBox(self, resolution, bFactor):

		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Resolution at 1% FDR-FSC: {:.2f}. \n \nEstimated B-factor: {:.2f}".format(resolution, bFactor));
		msg.setWindowTitle("Results");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	# ---------------------------------------------
	def runFSC(self):

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

		halfMap1Data = np.copy(half_map1.data);
		halfMap2Data = np.copy(half_map2.data);
		sizeMap = halfMap1Data.shape;

		# set output filename
		splitFilename = os.path.splitext(os.path.basename(self.fileLine_halfMap1.text()));
		outputFilename_PostProcessed = splitFilename[0] + "_postProcessed.mrc";


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

		#******************************************
		#*********** get num Asym Units ***********
		#******************************************

		try:
			numAsymUnits = int(self.numAsUnit.text());
		except:
			numAsymUnits = None;

		if numAsymUnits is not None:
			print('Using user provided number of asymmetric units, given as {:d}'.format(numAsymUnits));
		else:
			symmetry = self.symmetry.text();
			numAsymUnits = FSCutil.getNumAsymUnits(symmetry);
			print('Using provided ' + symmetry + ' symmetry. Number of asymmetric units: {:d}'.format(numAsymUnits));

		#******************************************
		#*************** get bFactor **************
		#******************************************
		try:
			bFactorInput = float(self.bFactor.text());
		except:
			bFactorInput = None;

		#run the FSC
		res, FSC, percentCutoffs, pValues, qValsFDR, resolution, _ = FSCutil.FSC(halfMap1Data, halfMap2Data,
																				 maskData, apix, 0.143,
																				 numAsymUnits, False, True, None,
																				 False);

		# write the FSC
		FSCutil.writeFSC(res, FSC, qValsFDR, pValues, resolution);

		if resolution < 8.0:

			# estimate b-factor and sharpen the map
			bFactor = FSCutil.estimateBfactor(0.5 * (halfMap1Data + halfMap2Data), resolution, apix, maskBFactor);

			if bFactorInput is not None:
				bFactor = bFactorInput;
				print('Using a user-specified B-factor of {:.2f} for map sharpening'.format(-bFactor));
			else:
				print('Using a B-factor of {:.2f} for map sharpening.'.format(-bFactor));

			processedMap = FDRutil.sharpenMap(0.5 * (halfMap1Data + halfMap2Data), -bFactor, apix, resolution);

			# write the post-processed map
			postProcMRC = mrcfile.new(outputFilename_PostProcessed, overwrite=True);
			postProc = np.float32(processedMap);
			postProcMRC.set_data(postProc);
			postProcMRC.voxel_size = apix;
			postProcMRC.close();

			output = "Saved sharpened and filtered map to: " + outputFilename_PostProcessed;
			print(output);

		end = time.time();
		totalRuntime = end - start;

		print("****** Summary ******");
		print("Runtime: %.2f" % totalRuntime);

		self.showMessageBox(resolution, bFactor);

