import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# *********************************
# ********* map-DW window *********
# *********************************

class MapDWWindow(QWidget):

	def __init__(self):
		super(MapDWWindow, self).__init__();
		layout = QFormLayout();

		# add input file
		hbox_maps = QHBoxLayout();
		self.fileLine_maps = QLineEdit();
		searchButton_maps = self.searchFileButton_maps();
		hbox_maps.addWidget(self.fileLine_maps);
		hbox_maps.addWidget(searchButton_maps);
		layout.addRow('EM maps', hbox_maps);

		self.apix = QLineEdit();
		layout.addRow('Pixel size [A]', self.apix);

		layout.addRow('',QHBoxLayout());
		layout.addRow('',QHBoxLayout());

		layout.addRow(' ', QHBoxLayout()); # make some space
		layout.addRow('Optional Input:', QHBoxLayout());

		self.lowPassFilter = QLineEdit();
		layout.addRow('Low-pass filter [A]', self.lowPassFilter);

		self.bFactor = QLineEdit();
		layout.addRow('B-factor', self.bFactor);

		self.addFrames = QLineEdit();
		layout.addRow('Binning of frames [#frames]', self.addFrames);

		self.skipFrames = QLineEdit();
		layout.addRow('Skip first frames [#frames]', self.skipFrames);


		# make some space
		layout.addRow('', QHBoxLayout());
		layout.addRow('', QHBoxLayout());

		# some buttons
		qtBtn = self.quitButton();
		runBtn = QPushButton('Run');
		runBtn.resize(runBtn.minimumSizeHint());


		buttonBox = QHBoxLayout();
		buttonBox.addWidget(qtBtn);
		buttonBox.addWidget(runBtn);

		layout.addRow(' ', buttonBox);
		self.setLayout(layout);

	def searchFileButton_maps(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_maps);
		return btn;

	def onInputFileButtonClicked_maps(self):
		filename = QFileDialog.getOpenFileNames(caption='Open file');

		if filename:
			self.fileLine_maps.setText(filename[0][0]);


	def quitButton(self):
		btn = QPushButton('Quit');
		btn.clicked.connect(QCoreApplication.instance().quit);
		btn.resize(btn.minimumSizeHint());

		return btn;

