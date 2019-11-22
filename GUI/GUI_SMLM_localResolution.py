from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from SMLMUtil import SMLM
from scipy import ndimage
import numpy as np
import time, os
import matplotlib.pyplot as plt

# ********************************
# ******* resolution window ******
# *********************************

class SMLMLocalResolutionWindow(QWidget):

	def __init__(self):
		super(SMLMLocalResolutionWindow, self).__init__();

		layout = QFormLayout();

		# ------------ now required input
		layout.addRow(' ', QHBoxLayout()); # make some space
		requiredLabel = QLabel("Required Input", self);
		requiredLabel.setFont(QFont('Arial', 17));
		layout.addRow(requiredLabel, QHBoxLayout());

		# add input file
		hbox_localizations = QHBoxLayout();
		self.fileLine_localizations = QLineEdit();
		searchButton_localizations = self.searchFileButton_localizations();
		hbox_localizations.addWidget(self.fileLine_localizations);
		hbox_localizations.addWidget(searchButton_localizations);
		layout.addRow('Localizations', hbox_localizations);

		#or
		orLabel = QLabel("or two half-images:", self);
		orLabel.setFont(QFont('Arial', 15));
		layout.addRow( '', orLabel);

		# ------- read two images
		# add input file
		hbox_image1 = QHBoxLayout();
		self.fileLine_image1 = QLineEdit();
		searchButton_image1 = self.searchFileButton_image1();
		hbox_image1.addWidget(self.fileLine_image1);
		hbox_image1.addWidget(searchButton_image1);
		layout.addRow('Image 1', hbox_image1);

		hbox_image2 = QHBoxLayout();
		self.fileLine_image2 = QLineEdit();
		searchButton_image2 = self.searchFileButton_image2();
		hbox_image2.addWidget(self.fileLine_image2);
		hbox_image2.addWidget(searchButton_image2);
		layout.addRow('Image 2', hbox_image2);

		#pixel size input
		self.apix = QLineEdit();
		self.apix.setText('2');
		layout.addRow('Pixel size to be used [nm]', self.apix);


		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());


		# window size input
		self.windowSize = QLineEdit();
		self.windowSize.setText('500');
		layout.addRow('Size of sliding window [pixels]', self.windowSize);


		# step size input
		self.stepSize = QLineEdit();
		self.stepSize.setText('200');
		layout.addRow('Step size of sliding window [pixels]', self.stepSize);

		# low resolution limit input
		self.lowRes = QLineEdit();
		self.lowRes.setText('None');
		layout.addRow('Low resolution limit [nm]', self.lowRes);

		# add output directory
		hbox_output = QHBoxLayout();
		self.fileLine_output = QLineEdit();
		searchButton_output = self.searchFileButton_output();
		hbox_output.addWidget(self.fileLine_output);
		hbox_output.addWidget(searchButton_output);
		layout.addRow('Save output to ', hbox_output);


		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.FSCBtn();

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);

		formGroupBox = QGroupBox();
		formGroupBox.setLayout(layout);

		#set the main Layout
		heading = QLabel("Global resolution estimation by FDR-FSC", self);
		heading.setFont(QFont('Arial', 17));
		heading.setAlignment(Qt.AlignTop);

		mainLayout = QVBoxLayout();
		mainLayout.addWidget(heading);
		mainLayout.addWidget(formGroupBox);
		mainLayout.addLayout(buttonBox);
		self.setLayout(mainLayout);

	#button localizations
	def searchFileButton_localizations(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_localizations);
		return btn;

	def onInputFileButtonClicked_localizations(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine_localizations.setText(filename[0]);

	#button image1
	def searchFileButton_image1(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_image1);
		return btn;

	def onInputFileButtonClicked_image1(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine_image1.setText(filename[0]);

	#button image2
	def searchFileButton_image2(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_image2);
		return btn;

	def onInputFileButtonClicked_image2(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine_image2.setText(filename[0]);

	#button output
	def searchFileButton_output(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_output);
		return btn;

	def onInputFileButtonClicked_output(self):
		filename = QFileDialog.getExistingDirectory(caption='Set output directory');
		if filename:
			self.fileLine_output.setText(filename);

	#button to quit
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


	def showMessageBox(self, path):

		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Local resolution estimation finished. Results saved to " + path);
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
		print('******* Significance analysis of FSC curves *******');
		print('***************************************************');


		# set working directory and output filename
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine_localizations.text());
		os.chdir(path);


		#read the input
		image1 = None;
		image2 = None;
		try:
			localizations = np.loadtxt(self.fileLine_localizations.text(), delimiter="	", skiprows=1, usecols=(4, 5));
		except:

			try:
				image1 = ndimage.imread(self.fileLine_image1.text());
				image2 = ndimage.imread(self.fileLine_image1.text());
				localizations = None;
			except:

				msg = QMessageBox();
				msg.setIcon(QMessageBox.Information);
				msg.setText("Cannot the input ...");
				msg.setWindowTitle("Error");
				msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
				retval = msg.exec_();
				return;

		#read the pixel size
		try:
			apix = float(self.apix.text());
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read pixel size ...");
			msg.setWindowTitle("Error");
			msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
			retval = msg.exec_();
			return;

		#read the window Size
		try:
			windowSize = int(self.windowSize.text());
		except:
			print("Window size needs to be a positive integer ...");
			return;

		#read the window Size
		try:
			stepSize = int(self.stepSize.text());
		except:
			print("Step size needs to be a positive integer ...");
			return;

		#read the low resolution limit
		try:
			lowResLimit = float(self.lowRes.text());
		except:
			lowResLimit = None;

		if lowResLimit is not None:
			print('Low resolution limit set to {:.2f} Angstroem.'.format(lowResLimit));
		else:
			print('No low resolution limit used ... ');

		#calcualte the actual local resolutions
		SMLMObject = SMLM.SMLM();
		SMLMObject.localResolution(localizations, image1, image2, apix, stepSize, windowSize, lowResLimit);

		# plot the local resolutions
		plt.imshow(SMLMObject.localResolutions.T, cmap='hot', origin='lower');
		#plt.colorbar();
		plt.savefig('localResolutions.png');
		plt.close();

		plt.imshow(SMLMObject.filteredMap.T, cmap='hot', origin='lower')
		#plt.colorbar();
		plt.savefig('heatMap_filt.png');
		plt.close();

		self.showMessageBox(path);

