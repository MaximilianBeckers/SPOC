from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from SMLMUtil import SMLM
import numpy as np
import time, os
import matplotlib.pyplot as plt

# ********************************
# ******* resolution window ******
# *********************************

class SMLMResolutionWindow(QWidget):

	def __init__(self):
		super(SMLMResolutionWindow, self).__init__();

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
		layout.addRow(orLabel, QHBoxLayout());


		self.apix = QLineEdit();
		self.apix.setText('2');
		layout.addRow('Pixel size to be used [nm]', self.apix);


		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());


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
		heading = QLabel("Global resolution estimation by FDR-FSC", self);
		heading.setFont(QFont('Arial', 15));
		heading.setAlignment(Qt.AlignTop);

		mainLayout = QVBoxLayout();
		mainLayout.addWidget(heading);
		mainLayout.addWidget(formGroupBox);
		mainLayout.addLayout(buttonBox);
		self.setLayout(mainLayout);

	def searchFileButton_localizations(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_localizations);
		return btn;

	def onInputFileButtonClicked_localizations(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine_localizations.setText(filename[0]);


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

		btn = QPushButton('Run FSC');
		btn.resize(btn.minimumSizeHint());
		btn.clicked.connect(self.runFSC);

		return btn;


	def showMessageBox(self, resolution):

		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Resolution at 1% FDR-FSC: {:.2f} nm".format(resolution));
		msg.setWindowTitle("Results");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	# ---------------------------------------------
	def runFSC(self):

		start = time.time();

		print('***************************************************');
		print('******* Significance analysis of FSC curves *******');
		print('***************************************************');


		# set working directory and output filename
		path = self.fileLine_output.text();
		if path == '':
			path = os.path.dirname(self.fileLine_localizations.text());
		os.chdir(path);


		#read the localizations
		try:
			localizations = np.loadtxt(self.fileLine_localizations.text(), delimiter="	", skiprows=1, usecols=(4, 5));
		except:
			msg = QMessageBox();
			msg.setIcon(QMessageBox.Information);
			msg.setText("Cannot read file ...");
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

		SMLMObject = SMLM.SMLM();
		SMLMObject.resolution(localizations, None, None, apix);

		plt.imshow(SMLMObject.fullMap.T, cmap='hot', origin='lower')
		plt.colorbar();
		plt.savefig('heatMap_full.png', dpi=300);
		plt.close();

		plt.imshow(SMLMObject.filteredMap.T, cmap='hot', origin='lower')
		plt.colorbar();
		plt.savefig('heatMap_filt.png', dpi=300);
		plt.close();

		self.showMessageBox(SMLMObject.resolution);
