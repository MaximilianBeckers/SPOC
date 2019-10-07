import numpy as np
from scipy.stats import ttest_1samp
import sys
import matplotlib.pyplot as plt
from FSCUtil import FSCutil, localResolutions2D
from confidenceMapUtil import FDRutil, mapUtil
import math

#****************************************************
class SMLM:

	embedding = [];
	lowRes = None;
	halfMap1 = [];
	halfMap2 = [];
	fullMap = [];
	localResolutions = [];
	filteredMap = [];
	dimension = 0;
	hannWindow = [];
	apix = 0.0;
	frequencyMap = [];
	FSCdata = [];
	resVec = [];
	qVals = [];
	resolution = 0.0;
	embeddings = [];


	#---------------------------------------------
	def resolution(self, embeddingData, image1, image2, apix):

		np.random.seed(3);

		self.apix = float(apix);

		#****************************
		#**** do two embeddings *****
		#****************************
		if embeddingData is not None:

			#split the localizations randomly in 2 half sets
			numLocalizations = embeddingData.shape[0];
			self.dimension = embeddingData.shape[1];
			sizeHalfSet = int(numLocalizations/2);
			permutedSequence = np.random.permutation(np.arange(numLocalizations));
			self.embeddingsHalf1 = embeddingData[permutedSequence[0:sizeHalfSet], :];
			self.embeddingsHalf2 = embeddingData[permutedSequence[sizeHalfSet:], :];
			self.embedding = embeddingData;
			self.make_half_maps();

		#****************************
		#***** use two images *******
		#****************************
		elif image1 is not None:

			self.halfMap1  = image1;
			self.halfMap2 = image2;

		self.hannWindow = FDRutil.makeHannWindow(self.halfMap1);
		maskData = self.hannWindow;

		self.fullMap = self.halfMap1 + self.halfMap2;
		self.frequencyMap = FSCutil.calculate_frequency_map(self.halfMap1);

		tmpResVec, FSC, _, _, qVals_FDR, resolution_FDR, _ = FSCutil.FSC(self.halfMap1, self.halfMap2, maskData, self.apix, 0.143, 1, False, True, None, True);

		self.resolution = resolution_FDR;
		self.FSCdata = FSC;
		#self.calcTTest();
		self.qVals = qVals_FDR;
		self.resVec = tmpResVec;
		self.filterMap();
		self.writeFSC();

	#---------------------------------------------
	def localResolution(self, embeddingData, image1, image2, apix, stepSize, boxSize, lowRes):

		np.random.seed(3);

		self.apix = float(apix);
		#****************************
		#**** do two embeddings *****
		#****************************
		if embeddingData is not None:

			#split the localizations randomly in 2 half sets
			numLocalizations = embeddingData.shape[0];
			self.dimension = embeddingData.shape[1];
			sizeHalfSet = int(numLocalizations/2);
			permutedSequence = np.random.permutation(np.arange(numLocalizations));
			self.embeddingsHalf1 = embeddingData[permutedSequence[0:sizeHalfSet], :];
			self.embeddingsHalf2 = embeddingData[permutedSequence[sizeHalfSet:], :];
			self.embedding = embeddingData;
			self.make_half_maps();

		#****************************
		#***** use two images *******
		#****************************
		elif image1 is not None:

			self.halfMap1  = image1;
			self.halfMap2 = image2;

		self.halfMap1 = self.halfMap1 + np.random.randn(self.halfMap1.shape[0], self.halfMap1.shape[1])*0.01;
		self.halfMap2 = self.halfMap2 + np.random.randn(self.halfMap2.shape[0], self.halfMap1.shape[1])*0.01;

		#maskData = FSCutil.makeCircularMask(self.halfMap1, (np.min(self.halfMap1.shape) / 2.0) - 4.0);  # circular mask
		maskData = np.ones(self.halfMap1.shape);

		self.fullMap = self.halfMap1 + self.halfMap2;
		self.frequencyMap = FSCutil.calculate_frequency_map(self.halfMap1);

		#estimate the local resolutions
		self.localResolutions = localResolutions2D.localResolutions2D(self.halfMap1, self.halfMap2, boxSize, stepSize, 0.5, self.apix, 1, maskData, maskData, lowRes);

		#do local filtering
		self.filteredMap, _, _, _ = mapUtil.localFiltration(self.fullMap, self.localResolutions, self.apix, False, None, None, None);
		self.filteredMap[self.filteredMap<0.0] = 0.0;

	#---------------------------------------------
	def make_half_maps(self):

		#*****************************************
		#**** if localizations are done in 2D ****
		#*****************************************
		if self.dimension == 2:

			#make the grid
			minX = min(np.amin(self.embeddingsHalf1[:,0]), np.amin(self.embeddingsHalf2[:,0]));
			maxX = max(np.amax(self.embeddingsHalf1[:,0]), np.amax(self.embeddingsHalf2[:,0]));
			minY = min(np.amin(self.embeddingsHalf1[:,1]), np.amin(self.embeddingsHalf2[:,1]));
			maxY = max(np.amax(self.embeddingsHalf1[:,1]), np.amax(self.embeddingsHalf2[:,1]));

			numPixX = (maxX-minX)/float(self.apix);
			numPixY = (maxY-minY)/float(self.apix);

			numPix = math.ceil(max(numPixX, numPixY));
			print("Images will have a size of " + repr(numPix) + "^2 ...");

			#initialize the half maps
			tmpHalfMap1 = np.zeros((numPix, numPix));
			tmpHalfMap2 = np.zeros((numPix, numPix));

			half1 = self.embeddingsHalf1;
			half2 = self.embeddingsHalf2;
                
			print("make halfmap 1 ...");
			#place localizations of HalfSet1
			for i in range(half1.shape[0]):

				#transform localization to the grid
				indicesInGrid = np.floor((half1[i, :] - np.array([minX, minY]))/self.apix);
				indicesInGrid = indicesInGrid.astype(int);
				tmpHalfMap1[indicesInGrid[0], indicesInGrid[1]] = tmpHalfMap1[indicesInGrid[0], indicesInGrid[1]] + 1.0;

			print("make halfmap 2 ...");
			#place localizations of HalfSet2
			for i in range(half2.shape[0]):

				#transform localization to the grid
				indicesInGrid = np.floor((half2[i, :] - np.array([minX, minY]))/self.apix);
				indicesInGrid = indicesInGrid.astype(int);
				tmpHalfMap2[indicesInGrid[0], indicesInGrid[1]] = tmpHalfMap2[indicesInGrid[0], indicesInGrid[1]] + 1.0;

		#*********************************************
		#****** if localizations are done in 3D ******
		#*********************************************

		elif self.dimension == 3:

			# make the grid
			minX = min(np.amin(self.embeddingsHalf1[:, 0]), np.amin(self.embeddingsHalf2[:, 0]));
			maxX = max(np.amax(self.embeddingsHalf1[:, 0]), np.amax(self.embeddingsHalf2[:, 0]));
			minY = min(np.amin(self.embeddingsHalf1[:, 1]), np.amin(self.embeddingsHalf2[:, 1]));
			maxY = max(np.amax(self.embeddingsHalf1[:, 1]), np.amax(self.embeddingsHalf2[:, 1]));
			minZ = min(np.amin(self.embeddingsHalf1[:, 2]), np.amin(self.embeddingsHalf2[:, 2]));
			maxZ = max(np.amax(self.embeddingsHalf1[:, 2]), np.amax(self.embeddingsHalf2[:, 2]));


			numPixX = (maxX - minX) / float(self.apix);
			numPixY = (maxY - minY) / float(self.apix);
			numPixZ = (maxZ - minZ) / float(self.apix);

			numPix = math.ceil(max(numPixX, numPixY, numPixZ));

			# initialize the half maps
			tmpHalfMap1 = np.zeros((numPix, numPix, numPix));
			tmpHalfMap2 = np.zeros((numPix, numPix, numPix));

			half1 = self.embeddingsHalf1;
			half2 = self.embeddingsHalf2;

			print("make halfmap 1 ...");
			# place localizations of HalfSet1
			for i in range(half1.shape[0]):
				# transform localization to the grid
				indicesInGrid = np.floor((half1[i, :] - np.array([minX, minY, minZ])) / self.apix);
				indicesInGrid = indicesInGrid.astype(int);
				tmpHalfMap1[indicesInGrid[0], indicesInGrid[1], indicesInGrid[2]] = tmpHalfMap1[indicesInGrid[0], indicesInGrid[1], indicesInGrid[2]] + 1.0;

			print("make halfmap 2 ...");
			# place localizations of HalfSet2
			for i in range(half2.shape[0]):
				# transform localization to the grid
				indicesInGrid = np.floor((half2[i, :] - np.array([minX, minY])) / self.apix);
				indicesInGrid = indicesInGrid.astype(int);
				tmpHalfMap2[indicesInGrid[0], indicesInGrid[1], indicesInGrid[2]] = tmpHalfMap2[indicesInGrid[0], indicesInGrid[1], indicesInGrid[2]] + 1.0;


		self.halfMap1 = tmpHalfMap1;
		self.halfMap2 = tmpHalfMap2;

	# --------------------------------------------
	def writeFSC(self):

		# *******************************
		# ******* write FSC plots *******
		# *******************************

		plt.plot(self.resVec, self.FSCdata, label="FSC", linewidth=1.5);

		# threshold the adjusted pValues
		self.qVals[self.qVals <= 0.01] = 0.0;

		plt.plot(self.resVec[0:][self.qVals == 0.0], self.qVals[self.qVals == 0.0] - 0.05, 'xr',
				 label="sign. at 1% FDR");

		plt.axhline(0.5, linewidth=0.5, color='r');
		plt.axhline(0.143, linewidth=0.5, color='r');
		plt.axhline(0.0, linewidth=0.5, color='b');
		plt.axvline(1.0/self.resolution, linewidth = 0.5, color='b');

		plt.xlabel("1/ Resolution");
		plt.ylabel("FSC");
		plt.legend();

		plt.savefig('FSC.png', dpi=300);
		plt.close();

    #---------------------------------------------
	def filterMap(self):

		if self.resolution != 0.0:
			#fourier transform the full map
			self.filteredMap = FDRutil.lowPassFilter(np.fft.rfftn(self.fullMap), self.frequencyMap, self.apix/float(self.resolution), self.fullMap.shape);
			self.filteredMap[self.filteredMap<0.0] = 0.0;

		else:
			self.filteredMap = np.zeros((10, 10));

	#---------------------------------------------
	def calcTTest(self):

		numShells = self.FSCdata.shape[0];
		sampleForTTest = self.FSCdata[int(0.8*numShells):];

		if sampleForTTest.shape[0] > 100:
			sampleForTTest = np.random.choice(sampleForTTest, 100);

		testResult = ttest_1samp(sampleForTTest, 0.0);
		pVal = testResult[1];

		if pVal<0.00001:
			print("FSC is significantly deviating from 0 at high-resolutions. Points are clustered too close, so sampling rate too low!")
			self.resolution = 0.00;
		else:
			print("Estimated resolution at 1% FDR-FSC: {:.3f}".format(self.resolution));

	#---------------------------------------------
	def makePlots(self):

		#plot all points
		fig, ax = plt.subplots(1, figsize=(14, 10));
		plt.scatter(*self.embedding.T, s=0.3, alpha=1.0);
		plt.title('all points');
		filename = "all_points.pdf";
		plt.savefig(filename, dpi=300);
		plt.close();

		#plot half1
		fig, ax = plt.subplots(1, figsize=(14, 10));
		plt.scatter(*self.embeddingsHalf1.T, s=0.3, alpha=1.0);
		plt.title('half1');
		filename = "half1.pdf";
		plt.savefig(filename, dpi=300);
		plt.close();


		#plot half2
		fig, ax = plt.subplots(1, figsize=(14, 10));
		plt.scatter(*self.embeddingsHalf2.T, s=0.3, alpha=1.0);
		plt.title('half2');
		filename = "half2.pdf";
		plt.savefig(filename, dpi=300);
		plt.close();


		#plot power spectrum of half1
		plt.imshow(  np.log(np.real(np.fft.fftshift(np.fft.fft2(self.halfMap1)))**2), cmap='Greys') ;
		plt.savefig("PS_half1.pdf", dpi=300);
		plt.close();


		#plot power spectrum of half2
		plt.imshow(  np.log(np.real(np.fft.fftshift(np.fft.fft2(self.halfMap2)))**2),cmap='Greys');
		plt.savefig("PS_half2.pdf", dpi=300);
		plt.close();
