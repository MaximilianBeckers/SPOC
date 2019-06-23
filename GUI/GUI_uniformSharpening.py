import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

# *********************************
# ****** sharpening window ********
# *********************************

class SharpeningWindow(QWidget):

	def __init__(self):
		super(SharpeningWindow, self).__init__();
		layout = QFormLayout();

		# add input file
		hbox_map = QHBoxLayout();
		self.fileLine = QLineEdit();
		searchButton_map = self.searchFileButton();
		hbox_map.addWidget(self.fileLine);
		hbox_map.addWidget(searchButton_map);
		layout.addRow('EM maps', hbox_map);

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

		hbox_FSC = QHBoxLayout();
		self.fileLine_FSC = QLineEdit();
		searchButton_FSC = self.searchFileButton_FSC();
		hbox_FSC.addWidget(self.fileLine_FSC);
		hbox_FSC.addWidget(searchButton_FSC);
		layout.addRow('FSC curve', hbox_FSC);

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

	def searchFileButton(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_map);
		return btn;

	def onInputFileButtonClicked_map(self):
		filename = QFileDialog.getOpenFileNames(caption='Open file');

		if filename:
			self.fileLine.setText(filename[0][0]);

	def searchFileButton_FSC(self):
		btn = QPushButton('Search File');
		btn.clicked.connect(self.onInputFileButtonClicked_FSC);
		return btn;

	def onInputFileButtonClicked_FSC(self):
		filename = QFileDialog.getOpenFileName(caption='Open file');

		if filename:
			self.fileLine.setText(filename[0]);

	def quitButton(self):
		btn = QPushButton('Quit');
		btn.clicked.connect(QCoreApplication.instance().quit);
		btn.resize(btn.minimumSizeHint());

		return btn;