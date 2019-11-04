import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from confidenceMapUtil import mapUtil, FDRutil
import mrcfile
import numpy as np
import time, os

# *********************************
# ****** sharpening window ********
# *********************************

class LocalFilteringWindow(QWidget):

	def __init__(self):
		super(LocalFilteringWindow, self).__init__();
		layout = QFormLayout();

		# ------------ now required input
		layout.addRow(' ', QHBoxLayout()); # make some space
		requiredLabel = QLabel("Required Input", self);
		requiredLabel.setFont(QFont('Arial', 17));
		layout.addRow(requiredLabel, QHBoxLayout());


		# add input file
		hbox_map = QHBoxLayout();
		self.fileLine = QLineEdit();
		searchButton_map = self.searchFileButton();
		hbox_map.addWidget(self.fileLine);
		hbox_map.addWidget(searchButton_map);
		layout.addRow('EM map', hbox_map);

		# add local resolution map
		hbox_locResMap = QHBoxLayout();
		self.fileLine_locResMap = QLineEdit();
		searchButton_locResMap = self.searchFileButton_locResMap();
		hbox_locResMap.addWidget(self.fileLine_locResMap);
		hbox_locResMap.addWidget(searchButton_locResMap);
		layout.addRow('Local resolution map', hbox_locResMap);

		layout.addRow('',QHBoxLayout());

		#--------------------------------------------------
		# ------------ now optional input -----------------
		#--------------------------------------------------
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());



		#------------------------------------
		#apix option
		self.apix = QLineEdit();
		self.apix.setText("None");
		layout.addRow('Pixel size [A]', self.apix);


		# add output directory
		hbox_output = QHBoxLayout();
		self.fileLine_output = QLineEdit();
		searchButton_output = self.searchFileButton_output();
		hbox_output.addWidget(self.fileLine_output);
		hbox_output.addWidget(searchButton_output);
		layout.addRow('Save output to ', hbox_output);


		#-----------------------------------
		#local normalization option
		self.localNormalization = QCheckBox(self);
		self.localNormalization.stateChanged.connect(self.localNormalizationState);
		layout.addRow('Do local background normalization?', self.localNormalization);

		# add box size for background noise estimation
		self.boxSize = QLineEdit();
		self.boxSize.setText('50');
		self.boxSizeText = QLabel('Size of window for background estimation [pixels]:');
		self.boxSize.setVisible(False);
		self.boxSizeText.setVisible(False)
		#layout.addRow('Size of window for background estimation [pixels]:', self.boxSize);
		layout.addRow(self.boxSizeText, self.boxSize);

		# add box coordinates
		self.coordBox = QHBoxLayout();
		self.xCoord = QLineEdit();
		self.yCoord = QLineEdit();
		self.zCoord = QLineEdit();
		self.xCoord.setText('None'), self.yCoord.setText('None'), self.zCoord.setText('None');
		self.coordBox.addWidget(self.xCoord);
		self.coordBox.addWidget(self.yCoord);
		self.coordBox.addWidget(self.zCoord);
		self.boxCoordText = QLabel('Box coordinates [x][y][z]:');
		self.boxCoordText.setVisible(False)
		self.xCoord.setVisible(False);
		self.yCoord.setVisible(False);
		self.zCoord.setVisible(False);
		layout.addRow(self.boxCoordText, self.coordBox);

		layout.addRow(' ', QHBoxLayout());  # make some space
		layout.addRow(' ', QHBoxLayout());  # make some space
		layout.addRow(' ', QHBoxLayout());  # make some space
		layout.addRow(' ', QHBoxLayout());  # make some space
		layout.addRow(' ', QHBoxLayout());  # make some space

		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.runBtn();
		self.checkNoiseBtn = self.checkNoiseEstimationBtn();
		self.checkNoiseBtn.setVisible(False);

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);
		buttonBox.addWidget(self.checkNoiseBtn)

		formGroupBox = QGroupBox();
		formGroupBox.setLayout(layout);

		#set the main Layout
		heading = QLabel("Local resolution filtering", self);
		heading.setFont(QFont('Arial', 17));
		heading.setAlignment(Qt.AlignTop)

		mainLayout = QVBoxLayout();
		mainLayout.addWidget(heading);
		mainLayout.addWidget(formGroupBox);
		mainLayout.addLayout(buttonBox);
		self.setLayout(mainLayout);


	def searchFileButton(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_map);
		return btn;

	def onInputFileButtonClicked_map(self):
		filename = QFileDialog.getOpenFileNames(caption='Open file');

		if filename:
			self.fileLine.setText(filename[0][0]);

	def searchFileButton_locResMap(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_locResMap);
		return btn;

	def onInputFileButtonClicked_locResMap(self):
		filename = QFileDialog.getOpenFileNames(caption='Open file');

		if filename:
			self.fileLine_locResMap.setText(filename[0][0]);

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

	def runBtn(self):
		btn = QPushButton('Run');
		btn.resize(btn.minimumSizeHint());
		btn.clicked.connect(self.runLocalFiltering);

		return btn;

	#--------------------------------------------------------
	def checkNoiseEstimationBtn(self):
		btn = QPushButton('Check Noise Estim.');
		btn.clicked.connect(self.checkNoiseEstimation);
		self.dialogs = list();
		btn.resize(btn.minimumSizeHint());

		return btn;

	def localNormalizationState(self):

		self.boxSizeText.setVisible(self.localNormalization.isChecked());
		self.boxSize.setVisible(self.localNormalization.isChecked());
		self.boxCoordText.setVisible(self.localNormalization.isChecked());
		self.xCoord.setVisible(self.localNormalization.isChecked())
		self.yCoord.setVisible(self.localNormalization.isChecked())
		self.zCoord.setVisible(self.localNormalization.isChecked())
		self.checkNoiseBtn.setVisible(self.localNormalization.isChecked())

	def showMessageBox(self):
		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Local resolution filtering finished!");
		msg.setWindowTitle("Finished");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	#---------------------------------------------------
	def runLocalFiltering(self):


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

		print('************************************************');
		print('**** Local resolution filtering of EM-Maps *****');
		print('************************************************');


		# read the maps
		try:
			em_map = mrcfile.open(self.fileLine.text(), mode='r');
			locResMap = mrcfile.open(self.fileLine_locResMap.text(), mode='r');
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		mapData = np.copy(em_map.data);
		locResMapData = np.copy(locResMap.data);

		# set working directory and output filename
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine.text());
		os.chdir(path);
		splitFilename = os.path.splitext(os.path.basename(self.fileLine.text()));
		outputFilename_locallyFiltered = splitFilename[0] + "_locallyFiltered.mrc";


		#**************************************
		#********* get pixel size *************
		#**************************************
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


		#**************************************
		#**** get noise estimation input ******
		#**************************************
		if self.localNormalization.isChecked():

			# ****************************************
			# ************ set the noiseBox **********
			# ****************************************
			try:
				boxCoord = [int(self.xCoord.text()), int(self.yCoord.text()), int(self.zCoord.text())];
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

			localVariance = True;
		else:
			localVariance = False;
			windowSize = None;
			boxCoord = None;



		#do the local filtering
		locFiltMap, meanMap, varMap, _ = mapUtil.localFiltration(mapData, locResMapData, apix, localVariance, windowSize,
															boxCoord, None);

		#if background normalization to be done, then do so
		if localVariance:
			locFiltMap = FDRutil.studentizeMap(locFiltMap, meanMap, varMap);


		# write the local resolution map
		localFiltMapMRC = mrcfile.new(outputFilename_locallyFiltered, overwrite=True);
		localFiltMap = np.float32(locFiltMap);
		localFiltMapMRC.set_data(localFiltMap);
		localFiltMapMRC.voxel_size = apix;
		localFiltMapMRC.close();

		end = time.time();
		totalRuntime = end - start;

		print("****** Summary ******");
		print("Runtime: %.2f" % totalRuntime);

		self.showMessageBox();

	# -----------------------------------------------------------
	# ------------------ run noise estima. check ----------------
	# -----------------------------------------------------------
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

		# set working directory and output filename
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine.text());
		os.chdir(path);

		try:
			windowSize = int(self.boxSize.text());
		except:
			print("Window size needs to be a positive integer ...")
			return;

		try:
			boxCoord = [int(self.xCoord.text()), int(self.yCoord.text()), int(self.zCoord.text())];
		except:
			boxCoord = 0;

		# generate the diagnostic image
		pp = mapUtil.makeDiagnosticPlot(mapData, windowSize, False, boxCoord);
		pp.savefig('diag_image.png');

		# now show the diagnostic image in new window
		dialog = NoiseWindow(self)
		self.dialogs.append(dialog);
		dialog.show();

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