import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from confidenceMapUtil import mapUtil
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
		layout.addRow('',QHBoxLayout());

		# ------------ now optional input
		layout.addRow(' ', QHBoxLayout()); # make some space
		optionLabel = QLabel("Optional Input", self);
		optionLabel.setFont(QFont('Arial', 17));
		layout.addRow(optionLabel, QHBoxLayout());

		self.apix = QLineEdit();
		self.apix.setText("None");
		layout.addRow('Pixel size [A]', self.apix);

		# make some space
		layout.addRow('', QHBoxLayout());
		layout.addRow('', QHBoxLayout());

		# some buttons
		qtBtn = self.quitButton();
		runBtn = self.runBtn();

		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);

		layout.addRow(' ', buttonBox);
		self.setLayout(layout);

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

	def showMessageBox(self):
		msg = QMessageBox();
		msg.setIcon(QMessageBox.Information);
		msg.setText("Local resolution filtering finished!");
		msg.setWindowTitle("Finished");
		msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
		retval = msg.exec_();

	#---------------------------------------------------
	def runLocalFiltering(self):

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

		# set output filename
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


		#do the local filtering
		locFiltMap, _, _, _ = mapUtil.localFiltration(mapData, locResMapData, apix, False, None,
															None, None);

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