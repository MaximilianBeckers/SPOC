import numpy as np
from confidenceMapUtil import FDRutil
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import math
import sys


#--------------------------------------------------------------
def calculate_frequency_map(map):

	#*********************************************************
	#*** calculation of the frequency map of the given map ***
	#*********************************************************

	sizeMap = map.shape;

	if map.ndim == 3:
		# calc frequency for each voxel
		freqi = np.fft.fftfreq(sizeMap[0], 1.0);
		freqj = np.fft.fftfreq(sizeMap[1], 1.0);
		freqk = np.fft.rfftfreq(sizeMap[2], 1.0);

		sizeFFT = np.array([freqi.size, freqj.size, freqk.size]);
		FFT = np.zeros(sizeFFT);

		freqMapi = np.copy(FFT);
		for j in range(sizeFFT[1]):
			for k in range(sizeFFT[2]):
				freqMapi[:, j, k] = freqi * freqi;

		freqMapj = np.copy(FFT);
		for i in range(sizeFFT[0]):
			for k in range(sizeFFT[2]):
				freqMapj[i, :, k] = freqj * freqj;

		freqMapk = np.copy(FFT);
		for i in range(sizeFFT[0]):
			for j in range(sizeFFT[1]):
				freqMapk[i, j, :] = freqk * freqk;

		frequencyMap = np.sqrt(freqMapi + freqMapj + freqMapk);

	elif map.ndim == 2:
		# calc frequency for each voxel
		freqi = np.fft.fftfreq(sizeMap[0], 1.0);
		freqj = np.fft.rfftfreq(sizeMap[1], 1.0);

		sizeFFT = np.array([freqi.size, freqj.size]);
		FFT = np.zeros(sizeFFT);

		freqMapi = np.copy(FFT);
		for j in range(sizeFFT[1]):
			freqMapi[:, j] = freqi * freqi;

		freqMapj = np.copy(FFT);
		for i in range(sizeFFT[0]):
			freqMapj[i, :] = freqj * freqj;

		frequencyMap = np.sqrt(freqMapi + freqMapj);

	return frequencyMap;

#--------------------------------------------------------------
def makeIndexVolumes(map):

	#***********************************************************
	#*** calculation of the angles for each voxel in the FFT ***
	#***********************************************************

	sizeMap = map.shape;

	if map.ndim == 3:
		# calc indices for each voxel
		indI = np.fft.fftfreq(sizeMap[0], 1.0)*sizeMap[0];
		indJ = np.fft.fftfreq(sizeMap[1], 1.0)*sizeMap[1];
		indK = np.fft.fftfreq(sizeMap[2], 1.0)*sizeMap[2];

		sizeFFT = np.array([indI.size, indJ.size, indK.size]);
		FFT = np.zeros(sizeFFT);

		indMapI = np.copy(FFT);
		for j in range(sizeFFT[1]):
			for k in range(sizeFFT[2]):
				indMapI[:, j, k] = indI;

		indMapJ = np.copy(FFT);
		for i in range(sizeFFT[0]):
			for k in range(sizeFFT[2]):
				indMapJ[i, :, k] = indJ;

		indMapK = np.copy(FFT);
		for i in range(sizeFFT[0]):
			for j in range(sizeFFT[1]):
				indMapK[i, j, :] = indK;


	elif map.ndim == 2:
		# calc frequency for each voxel
		indI = np.fft.fftfreq(sizeMap[0], 1.0);
		indJ = np.fft.rfftfreq(sizeMap[1], 1.0);

		sizeFFT = np.array([indI.size, indJ.size]);
		FFT = np.zeros(sizeFFT);

		indMapI = np.copy(FFT);
		for j in range(sizeFFT[1]):
			indMapI[:, j] = indI;

		indMapJ = np.copy(FFT);
		for i in range(sizeFFT[0]):
			indMapJ[i, :] = indJ;


	return  indMapI, indMapJ, indMapK;

#--------------------------------------------------------------
def makeDirResVolumes(map, dirResMap):

	#***********************************************************
	#*** calculation of the angles for each voxel in the FFT ***
	#***********************************************************

	sizeMap = map.shape;


	#first calculate the indices for voxel in the fft
	if map.ndim == 3:
		# calc indices for each voxel
		indI = np.around(np.fft.fftfreq(sizeMap[0], 1.0)*sizeMap[0]);
		indJ = np.around(np.fft.fftfreq(sizeMap[1], 1.0)*sizeMap[1]);
		indK = np.around(np.fft.rfftfreq(sizeMap[2], 1.0)*sizeMap[2]);

		sizeFFT = np.array([indI.size, indJ.size, indK.size]);
		FFT = np.zeros(sizeFFT);

		indMapI = np.copy(FFT);
		for j in range(sizeFFT[1]):
			for k in range(sizeFFT[2]):
				indMapI[:, j, k] = indI;

		indMapJ = np.copy(FFT);
		for i in range(sizeFFT[0]):
			for k in range(sizeFFT[2]):
				indMapJ[i, :, k] = indJ;

		indMapK = np.copy(FFT);
		for i in range(sizeFFT[0]):
			for j in range(sizeFFT[1]):
				indMapK[i, j, :] = indK;

		radiusMap = np.sqrt(indMapI**2 + indMapJ**2);

		azimuth = np.arctan2(indMapJ, indMapI) * 180.0/np.pi;
		el = np.arctan2(indMapK, radiusMap) * 180.0/np.pi;

	x = np.linspace(1, 10, dirResMap.shape[0]);
	y = np.linspace(1, 10, dirResMap.shape[1]);
	myInterpolatingFunction = RegularGridInterpolator((x, y), dirResMap, method='linear');
	xNew = np.linspace(1, 10, 361);
	yNew = np.linspace(1, 10, 91);
	xInd, yInd = np.meshgrid(xNew, yNew, indexing='ij', sparse=True);
	dirResMap = myInterpolatingFunction((xInd, yInd));

	dirResMap3D = np.zeros(sizeMap);

	shiftX =  np.fft.rfftfreq(sizeMap[0], 1.0).size - 1;
	shiftY =  np.fft.rfftfreq(sizeMap[1], 1.0).size - 1;
	shiftZ =  np.fft.rfftfreq(sizeMap[2], 1.0).size - 2;


	for i in range(sizeFFT[0]):
		for j in range(sizeFFT[1]):
			for k in range(sizeFFT[2]):

				x = int(indI[i]) + shiftX;
				y = int(indJ[j]) + shiftY;
				z = int(indK[k]) + shiftZ;

				phi = int(azimuth[i,j,k]);
				theta = int(el[i,j,k]);

				dirResMap3D[x, y, z] = dirResMap[int(math.floor(phi)), int(math.floor(theta))]


	dirResMap3D[:,:,0:shiftZ] = np.flip(dirResMap3D[:,:,shiftZ:2*shiftZ]);

	return dirResMap3D;

#---------------------------------------------------------------------------------
def makeCircularMask(map, sphereRadius):

	#***********************************************************
	#************** calculate spherical mask-map ***************
	#*** with sizei of the given map and radius sphereRadius ***
	#***********************************************************

	#some initialization
	mapSize = map.shape;

	if map.ndim == 3:
		x = np.linspace(-math.floor(mapSize[0]/2.0), -math.floor(mapSize[0]/2.0) + mapSize[0], mapSize[0]);
		y = np.linspace(-math.floor(mapSize[1]/2.0), -math.floor(mapSize[1]/2.0) + mapSize[1], mapSize[1]);
		z = np.linspace(-math.floor(mapSize[2]/2.0), -math.floor(mapSize[2]/2.0) + mapSize[2], mapSize[2]);

		xx, yy, zz = np.meshgrid(x, y, z, indexing='ij');

		radiusMap = np.sqrt(xx**2 + yy**2 + zz**2);

	elif map.ndim == 2:
		x = np.linspace(-math.floor(mapSize[0]/2.0), -math.floor(mapSize[0]/2.0) + mapSize[0], mapSize[0]);
		y = np.linspace(-math.floor(mapSize[1]/2.0), -math.floor(mapSize[1]/2.0) + mapSize[1], mapSize[1]);

		xx, yy = np.meshgrid(x, y, indexing='ij');

		radiusMap = np.sqrt(xx**2 + yy**2);


	#now extend mask with some smooth fall off
	gaussFallOffSigma = 2;

	tmpRadiusMap = radiusMap-sphereRadius;
	fallOffMap = np.exp(-((tmpRadiusMap)**2)/(2.0*gaussFallOffSigma**2));
	fallOffMap[fallOffMap < 0.000001] = 0.0;

	mask = fallOffMap;
	mask[tmpRadiusMap<0.0] = 1.0;

	return mask;

#---------------------------------------------------------------------------------
def estimateBfactor(map, resolution, apix, maskData):

	#*************************************
	#***** estimate B-factor falloff *****
	#*************************************

	sizeMap = map.shape;

	#calculate the frequency map
	freqMap = calculate_frequency_map(map);
	freqMap = freqMap/float(apix);

	#do FFT
	try:
		import pyfftw
		import multiprocessing

		numCores = multiprocessing.cpu_count();
		fftObject = pyfftw.builders.rfftn(maskData, threads=numCores);
		FFTmap = fftObject(map*maskData);
	except:
		FFTmap = np.fft.rfftn(map*maskData)

	res = np.fft.rfftfreq(sizeMap[0], 1.0);
	res = res / float(apix);

	resSquared = res*res;

	#get rotationally averaged power spectrum
	F = getRotAvgStucFac(res, freqMap, FFTmap);
	lnF = np.log(F);

	#do linear regression alpha+beta*x for resolutions better than 10 Angstroem
	subset_lnF = lnF[(res>(1.0/10.0)) & (res<(1.0/float(resolution)))];
	subset_res = resSquared[(res>(1.0/10.0)) & (res<(1.0/float(resolution)))];

	bFactor = (np.sum(subset_lnF*subset_res) - (1.0/float(subset_res.size))*np.sum(subset_lnF)*np.sum(subset_res))/(np.sum(np.square(subset_res))- (1.0/subset_res.size)*(np.square(np.sum(subset_res))));
	alpha = np.mean(subset_lnF) - bFactor*np.mean(subset_res);

	#make Guinier plot
	plt.plot(resSquared, lnF, label="Guinier Plot", linewidth=1.5);
	plt.xlabel("1/resolution^2 [1/A^2] ");
	plt.ylabel("log(|F|)");
	plt.savefig("GuinierPlot.pdf", dpi=300);
	plt.close();

	output = "Estimated B-factor of the provided map: %.2f" % (-4.0*bFactor);
	print(output);

	return -4.0*bFactor;

#------------------------------------------------------
def getRotAvgStucFac(res, freqMap, FFTmap):

	#********************************************
	#******* calculate rotationally avg. ********
	#************* power spectrum ***************
	#********************************************

	numRes = res.shape[0];
	resSpacing = (res[1] - res[0]) / 2.0;
	F = np.copy(res);

	#generate data for guinier plot
	for i in range(numRes):

		tmpRes = res[i];
		rotAvgStrucFac = np.mean(np.absolute(FFTmap[((tmpRes - resSpacing) < freqMap) & (freqMap <= (tmpRes + resSpacing))]));
		F[i] = rotAvgStrucFac;

	return F;

#--------------------------------------------------------
def FSC(halfMap1, halfMap2, maskData, apix, cutoff, numAsymUnits, localRes, verbose, permutedCorCoeffs, SMLM):

	#********************************************
	#***** function that calculates the FSC *****
	#********************************************

	if localRes:
		maskCoeff = 0.23;
	elif SMLM:
		maskCoeff = 0.6;
	else:
		maskCoeff = 0.7;

	if maskData is not None:
		halfMap1 = halfMap1*maskData;
		halfMap2 = halfMap2*maskData;

	#calculate frequency map
	freqMap = calculate_frequency_map(halfMap1);
	freqMap = freqMap/float(apix);

	#do fourier transforms
	try:
		import pyfftw
		import multiprocessing

		fftObject_half1 = pyfftw.builders.rfftn(halfMap1);
		fftObject_half2 = pyfftw.builders.rfftn(halfMap2);
		fft_half1 = fftObject_half1(halfMap1);
		fft_half2 = fftObject_half2(halfMap2);
	except:
		fft_half1 = np.fft.rfftn(halfMap1);
		fft_half2 = np.fft.rfftn(halfMap2);

	sizeMap = halfMap1.shape;

	res = np.fft.rfftfreq(sizeMap[0], 1.0);
	res = res/float(apix);
	numRes = res.shape[0];

	resSpacing = (res[1] - res[0])/2.0;
	FSC = np.ones((res.shape[0]));
	pVals = np.zeros((res.shape[0]));
	percentCutoffs = np.zeros((res.shape[0], 4));
	threeSigma = np.zeros((res.shape[0]));
	threeSigmaCorr = np.zeros((res.shape[0]));
	tmpPermutedCorCoeffs = [];

	numCalculations = res.shape[0];

	if verbose:
		print("Run permutation test of each resolution shell ...");

	for i in range(res.shape[0]):
		tmpRes = res[i];
		resShell_half1 = fft_half1[((tmpRes - resSpacing) < freqMap) & (freqMap < (tmpRes + resSpacing))];
		resShell_half2 = fft_half2[((tmpRes - resSpacing) < freqMap) & (freqMap < (tmpRes + resSpacing))];

		FSC[i] = correlationCoefficient(resShell_half1, resShell_half2);

		if (permutedCorCoeffs is not None): #for local resolution estimation
			tmpCorCoeffs = permutedCorCoeffs[i];
			pVals[i] = (tmpCorCoeffs[tmpCorCoeffs > FSC[i]].shape[0])/(float(tmpCorCoeffs.shape[0]));
			tmpPermutedCorCoeffs = None;
		else:
			pVals[i], percentCutoffs[i,:], threeSigma[i], threeSigmaCorr[i], corCoeffs = permutationTest(resShell_half1, resShell_half2, numAsymUnits, maskCoeff);
			tmpPermutedCorCoeffs.append(corCoeffs);

		if verbose:
			#print output
			progress = i/float(numCalculations);
			if i%(int(numCalculations/20.0)) == 0:
				output = "%.1f" %(progress*100) + "% finished ..." ;
				print(output);
		
	pVals[0] = 0.0;

	#for the first two resolutions shells, use a 0.75 FSC criterion for local resolutions, as permutation not reliabele for such small sample sizes
	if localRes or SMLM:
		if FSC[0] < 0.75:
			pVals[0] = 1.0;
		else:
			pVals[0] = 0.0;
		
		if FSC[1] < 0.75:
			pVals[1] = 1.0;
		else:
			pVals[1] = 0.0;

	# do FDR control of p-Values
	qVals_FDR = FDRutil.pAdjust(pVals, 'BY');

	tmpFSC = np.copy(FSC);
	tmpFSC[tmpFSC > cutoff] = 1.0;
	tmpFSC[tmpFSC <= cutoff] = 0.0;
	tmpFSC = 1.0 - tmpFSC;
	tmpFSC[0] = 0.0;	
	tmpFSC[1] = 0.0;

	try:
		resolution = np.min(np.argwhere(tmpFSC))-1;

		if resolution < 0:
			resolution = 0.0;
		else:
			if res[int(resolution)] == 0.0:
				resolution = 0.0;
			else:
				tmpFreq = res[int(resolution)] #+ (res[resolution+1] - res[resolution])/2.0;
				resolution = float(1.0/tmpFreq);
	except:
		resolution = 2.0*apix;

	threshQVals = np.copy(qVals_FDR);
	threshQVals[threshQVals <= 0.01] = 0.0; #signal
	threshQVals[threshQVals > 0.01] = 1.0 #no signal

	try:
		resolution_FDR = np.min(np.argwhere(threshQVals)) - 1;

		if resolution_FDR < 0:
			resolution_FDR = 0.0;
		else:
			if res[int(resolution_FDR)] == 0.0:
				resolution_FDR = 0.0;
			else:
				tmpFreq = res[int(resolution_FDR)] #+ (res[resolution_FDR + 1] - res[resolution_FDR]) / 2.0;
				resolution_FDR = float(1.0/tmpFreq);
	except:
		resolution_FDR = 2.0*apix;

	if verbose:
		print('Resolution at a unmasked ' + repr(cutoff) + ' FSC threshold: ' + repr(round(resolution, 2)));
		print('Resolution at 1 % FDR-FSC: ' + repr(round(resolution_FDR, 2)) + ' Angstrom');
		#print('Resolution at 0.01 % FDR: ' + repr(round(resolution_FDR01, 2)) + ' Angstrom');
		#print('Resolution at 1 % FWER: ' + repr(round(resolution_FWER, 2)) + ' Angstrom');

	return res, FSC, percentCutoffs, pVals, qVals_FDR, resolution_FDR, tmpPermutedCorCoeffs;


#--------------------------------------------------------
def threeDimensionalFSC(halfMap1, halfMap2, maskData, apix, cutoff, numAsymUnits, samplingAzimuth, samplingElevation, coneOpening):

	#***********************************************
	#***** function that calculates the 3D FSC *****
	#***********************************************

	maskCoeff = 0.7;

	if maskData is not None:
		halfMap1 = halfMap1*maskData;
		halfMap2 = halfMap2*maskData;

	sizeMap = halfMap1.shape;



	# calc frequency for each voxel
	freqi = np.fft.fftfreq(sizeMap[0], 1.0);
	freqj = np.fft.fftfreq(sizeMap[1], 1.0);
	freqk = np.fft.fftfreq(sizeMap[2], 1.0);

	sizeFFT = np.array([freqi.size, freqj.size, freqk.size]);
	FFT = np.zeros(sizeFFT);

	freqMapi = np.copy(FFT);
	for j in range(sizeFFT[1]):
		for k in range(sizeFFT[2]):
			freqMapi[:, j, k] = freqi * freqi;

	freqMapj = np.copy(FFT);
	for i in range(sizeFFT[0]):
		for k in range(sizeFFT[2]):
			freqMapj[i, :, k] = freqj * freqj;

	freqMapk = np.copy(FFT);
	for i in range(sizeFFT[0]):
		for j in range(sizeFFT[1]):
			freqMapk[i, j, :] = freqk * freqk;

	freqMap = np.sqrt(freqMapi + freqMapj + freqMapk);
	freqMap = freqMap / float(apix);

	#calculate spherical coordinates for each voxel in FFT
	xInd, yInd, zInd = makeIndexVolumes(halfMap1);


	# do fourier transforms
	try:
		import pyfftw
		import multiprocessing

		fftObject_half1 = pyfftw.builders.fftn(halfMap1);
		fftObject_half2 = pyfftw.builders.fftn(halfMap2);
		fft_half1 = fftObject_half1(halfMap1);
		fft_half2 = fftObject_half2(halfMap2);
	except:
		fft_half1 = np.fft.fftn(halfMap1);
		fft_half2 = np.fft.fftn(halfMap2);

	res = np.fft.rfftfreq(sizeMap[0], 1.0);
	res = res/float(apix);
	numRes = res.shape[0];

	resSpacing = (res[1] - res[0])/2.0;

	#initialize data
	FSC = np.ones((res.shape[0]));
	pVals = np.zeros((res.shape[0]));
	directionalResolutions = [];
	permutedCorCoeffs = [[]];
	phiArray = [];
	thetaArray = [];
	angleSpacing = (coneOpening/360.0)*2*np.pi; #correspond to 20 degrees
	directionalResolutionMap = np.zeros((samplingAzimuth, samplingElevation));

	phiAngles = np.linspace(-np.pi, np.pi, samplingAzimuth);
	thetaAngles = np.linspace(0, (np.pi/2.0), samplingElevation);

	#phi is azimuth (-pi, pi), theta is polar angle (0,pi), 100 sampling points each
	for phiIndex in range(samplingAzimuth):

		phi = phiAngles[phiIndex];

		for thetaIndex in range(samplingElevation):

			theta = thetaAngles[thetaIndex];

			#get point with the specified angles and radius 1
			x = np.cos(theta)*np.cos(phi);
			y = np.cos(theta)*np.sin(phi);
			z = np.sin(theta);

			#get angle to (x,y,z) for all points in the map
			with np.errstate(divide='ignore', invalid='ignore'):
				angles = np.arccos(np.divide(x*xInd + y*yInd + z*zInd, (np.sqrt(xInd**2 + yInd**2 + zInd**2)*np.sqrt(x**2+y**2+z**2))));
				angles[~ np.isfinite(angles)] = 0;

			for i in range(res.shape[0]):

				tmpRes = res[i];

				currentIndices = (((tmpRes - resSpacing) < freqMap) & (freqMap < (tmpRes + resSpacing)) & (np.absolute(angles) < angleSpacing));

				resShell_half1 = fft_half1[currentIndices];
				resShell_half2 = fft_half2[currentIndices];

				if (resShell_half1.size < 10) and (thetaIndex==0) and (phiIndex==0): #if no samples in this shell, accept it

					permutedCorCoeffs.append([]);
					pVals[i] = 0.0;
					FSC[i] = 1.0;

				else:
					FSC[i] = correlationCoefficient(resShell_half1, resShell_half2);

					if (thetaIndex==0) and (phiIndex==0):
						pVals[i], _, _, _, tmpPermutedCorCoeffs = permutationTest(resShell_half1, resShell_half2, numAsymUnits, maskCoeff);
						permutedCorCoeffs.append(np.copy(tmpPermutedCorCoeffs));
					else:
						tmpCorCoeffs = permutedCorCoeffs[i];
						if tmpCorCoeffs == []:
							pVals[i] = 0.0;
						else:
							pVals[i] = (tmpCorCoeffs[tmpCorCoeffs > FSC[i]].shape[0]) / (float(tmpCorCoeffs.shape[0]));


			# do FDR control of p-Values
			qVals_FDR = FDRutil.pAdjust(pVals, 'BY');

			tmpFSC = np.copy(FSC);
			tmpFSC[tmpFSC > cutoff] = 1.0;
			tmpFSC[tmpFSC <= cutoff] = 0.0;
			tmpFSC = 1.0 - tmpFSC;
			tmpFSC[0] = 0.0;
			tmpFSC[1] = 0.0;

			print(tmpFSC);

			try:
				resolution = np.min(np.argwhere(tmpFSC)) - 1;

				if resolution < 0:
					resolution = 0.0;
				else:
					if res[int(resolution)] == 0.0:
						resolution = 0.0;
					else:
						tmpFreq = res[int(resolution)]  # + (res[resolution+1] - res[resolution])/2.0;
						resolution = float(1.0 / tmpFreq);
			except:
				resolution = 2.0 * apix;

			threshQVals = np.copy(qVals_FDR);
			threshQVals[threshQVals <= 0.01] = 0.0;  # signal
			threshQVals[threshQVals > 0.01] = 1.0  # no signal

			try:
				resolution_FDR = np.min(np.argwhere(threshQVals)) - 1;

				if resolution_FDR < 0:
					resolution_FDR = 0.0;
				else:
					if res[int(resolution_FDR)] == 0.0:
						resolution_FDR = 0.0;
					else:
						tmpFreq = res[int(resolution_FDR)]  # + (res[resolution_FDR + 1] - res[resolution_FDR]) / 2.0;
						resolution_FDR = float(1.0 / tmpFreq);
			except:
				resolution_FDR = 2.0 * apix;

			#append the resolutions
			directionalResolutionMap[phiIndex, thetaIndex] = resolution_FDR;
			np.append(directionalResolutions, resolution_FDR);
			np.append(phiArray, phi);
			np.append(thetaArray, theta);


		# print progress
		progress = (phiIndex+1) / float(samplingAzimuth);
		if phiIndex % (int(math.ceil(samplingAzimuth / 20.0))) == 0:
			output = "%.1f" % (progress * 100) + "% finished ...";
			print(output);

	# *************************************
	# ********** do interpolation *********
	# *************************************
	print("Interpolating directional Resolutions ...");
	x = np.linspace(1, 10, samplingAzimuth);
	y = np.linspace(1, 10, samplingElevation);
	myInterpolatingFunction = RegularGridInterpolator((x, y), directionalResolutionMap, method='linear')
	xNew = np.linspace(1, 10, 15*samplingAzimuth);
	yNew = np.linspace(1, 10, 15*samplingElevation);
	xInd, yInd = np.meshgrid(xNew, yNew, indexing='ij', sparse=True);
	directionalResolutionMap = myInterpolatingFunction((xInd, yInd));

	dirResMap = makeDirResVolumes(halfMap1, directionalResolutionMap);

	return phiArray, thetaArray, directionalResolutions, directionalResolutionMap, dirResMap;

#--------------------------------------------------------
def correlationCoefficient(sample1, sample2):

	#*******************************
	#*** calc. correlation coeff ***
	#*******************************

	FSCnominator = np.sum((sample1 * np.conj(sample2)) + (np.conj(sample1) * sample2));
	FSCdenominator = np.sqrt(np.sum(2.0*np.square(np.absolute(sample1))) * 2.0*np.sum(np.square(np.absolute(sample2))));

	if FSCdenominator != 0:
		FSC = np.real(FSCnominator) / np.real(FSCdenominator);
	else:
		FSC = 0.0;

	return FSC;

#--------------------------------------------------------
def permutationTest(sample1, sample2, numAsymUnits, maskCoeff):

	#***************************************************
	#**** permutation-test of the two given samples ****
	#***************************************************

	#now get effective sample sizes
	numSamples = sample2.shape[0];
	maxSamples = np.min(((maskCoeff)*numSamples/float(numAsymUnits), 200000));
	maxSamples = np.max((maxSamples, 1.0));
	maxSamples = int(maxSamples);
	numSamplesThreeSigma = np.min((np.max((numSamples/float(numAsymUnits), 1.0)), 200000));
	numSamplesThreeSigma = int(numSamplesThreeSigma);

	#effective sample size van Heel (2005)
	numSamplesThreeSigmaCorr = np.min((np.max((numSamples/float(numAsymUnits)*((3.0/2.0)*maskCoeff)**2, 1.0)), 200000));
	#numSamplesThreeSigmaCorr = np.min((np.max((numSamples/float(numAsymUnits)*maskCoeff, 1.0)), 200000));
	numSamplesThreeSigmaCorr = int(numSamplesThreeSigmaCorr);

	#get real FSC value
	trueFSC = correlationCoefficient(sample1, sample2);

	numPermutations = np.min((math.factorial(numSamples), 1000));
	corrCoeff = np.zeros(numPermutations);

	#set random seed
	np.random.seed(3);

	if numSamples > maxSamples:  # do subsampling
		randomIndices = np.random.choice(range(numSamples), maxSamples, replace=False);
		tmpSample1 = np.copy(sample1[randomIndices]);
		tmpSample2 = np.copy(sample2[randomIndices]);
		numSamples = maxSamples;
	else:
		tmpSample1 = np.copy(sample1);
		tmpSample2 = np.copy(sample2);

	tmpFSCdenominator = np.sqrt(np.sum(2.0 * np.square(np.absolute(tmpSample1))) * 2.0 * np.sum(np.square(np.absolute(tmpSample2))));
	tmpSample1ComplexConj = np.conj(tmpSample1);
	tmpSample1 = 0; #free memory

	#do the actual permutations
	permutedCorCoeffs, numPermutations = doPermutations(tmpSample2, tmpSample1ComplexConj, numPermutations, tmpFSCdenominator, trueFSC);

	permutedCorCoeffs = np.real(permutedCorCoeffs)/np.real(tmpFSCdenominator);

	#calculate the pValue
	extremeCorCoeff = permutedCorCoeffs[permutedCorCoeffs>trueFSC];
	numExtreme = extremeCorCoeff.shape[0];
	pValue = numExtreme/float(numPermutations);

	#plot 3Sigma curve
	threeSigma = 3.0/(math.sqrt(numSamplesThreeSigma)+3.0-1.0);
	threeSigmaCorr = 3.0/(math.sqrt(numSamplesThreeSigmaCorr)+3.0-1.0);

	#calculate one percent cutoffs, i.e. cutoffs for each voxel before multiple testing correction
	permutedCorCoeffs = np.sort(permutedCorCoeffs);
	_onePercentCutoff = permutedCorCoeffs[int(numPermutations - math.floor(numPermutations * 0.0015)) - 1];
	onePercentCutoff = permutedCorCoeffs[int(numPermutations - math.floor(numPermutations * 0.01)) - 1];
	fivePercentCutoff = permutedCorCoeffs[int(numPermutations - math.floor(numPermutations * 0.05)) - 1];
	tenPercentCutoff = permutedCorCoeffs[int(numPermutations - math.floor(numPermutations * 0.10)) - 1];
	percentCutoffs = np.array((tenPercentCutoff, fivePercentCutoff, onePercentCutoff, _onePercentCutoff));
	percentCutoffs[percentCutoffs<=0] = 1.0;

	#if sample size of FSC values is low, use a 0.75 FSC threshold
	if maxSamples < 10:
		if trueFSC < 0.75:
			pValue = 1.0;
		else:
			pValue = 0.0;

		percentCutoffs = np.ones(percentCutoffs.shape);

	return pValue, percentCutoffs, threeSigma, threeSigmaCorr, permutedCorCoeffs;

#--------------------------------------------------------
def doPermutations(tmpSample2, tmpSample1ComplexConj, numPermutations, tmpFSCdenominator, trueFSC):


	prevPValue = 0.0;
	permutedCorCoeffs = np.zeros(0);
	itNumPermutations = 200;  #check p-value every 200 permutations

	if numPermutations<1000: #for the first resolutions shells no p-value convergence test

		tmpPermutedCorCoeffs = generatePermutations(numPermutations, tmpSample1ComplexConj, tmpSample2);
		permutedCorCoeffs = np.append(permutedCorCoeffs, tmpPermutedCorCoeffs);

	else: #check convergence of p-values every 200 permutations

		numChecks = int(numPermutations/float(itNumPermutations));

		for i in range(numChecks):

			tmpPermutedCorCoeffs = generatePermutations(itNumPermutations, tmpSample1ComplexConj, tmpSample2);
			permutedCorCoeffs = np.append(permutedCorCoeffs, tmpPermutedCorCoeffs);

			#calculate temporary p-value
			tmpPermutedCorCoeffs = np.real(permutedCorCoeffs) / np.real(tmpFSCdenominator);

			#calculate the pValue
			extremeCorCoeff = tmpPermutedCorCoeffs[tmpPermutedCorCoeffs>trueFSC];
			numExtreme = extremeCorCoeff.shape[0];
			newPValue = numExtreme / float(tmpPermutedCorCoeffs.size);

			# if p-value accuracy is high enough, stop permutations
			if (newPValue - prevPValue) < 0.001:
				numPermutations = (i + 1)*itNumPermutations;
				break;

			prevPValue = newPValue;

	return permutedCorCoeffs, numPermutations;

#-------------------------------------------------------
def generatePermutations(numPermutations, tmpSample1ComplexConj, tmpSample2):

	permutedCorCoeffs = np.zeros(numPermutations);
	for i in range(numPermutations):

		permutedSample2 = np.random.permutation(tmpSample2);

		summand = tmpSample1ComplexConj * permutedSample2;
		tmpCorCoeff = np.sum(np.conj(summand) + summand);

		permutedCorCoeffs[i] = np.real(tmpCorCoeff);

	return permutedCorCoeffs;

#--------------------------------------------------------
def writeFSC(resolutions, FSC, qValuesFDR, pValues, resolution):

	#*******************************
	#******* write FSC plots *******
	#*******************************

	plt.plot(resolutions, FSC, label="FSC", linewidth=1.5);

	#threshold the adjusted pValues
	qValuesFDR[qValuesFDR<=0.01] = 0.0;
	pValues[pValues<=0.01] = 0.0;

	plt.plot(resolutions[0:][qValuesFDR==0.0], qValuesFDR[qValuesFDR==0.0]-0.05, 'xr', label="sign. at 1% FDR");
	#plt.plot(resolutions[0:][pValues==0.0], pValues[pValues==0.0]-0.1, 'xb', label="sign. at 1%");

	plt.axhline(0.5, linewidth = 0.5, color = 'r');
	plt.axhline(0.143, linewidth = 0.5, color = 'r');
	plt.axhline(0.0, linewidth = 0.5, color = 'b');
	plt.axvline(1.0/resolution, linewidth = 0.5, color = 'b');

	#add text with resolution
	plt.text(0, 1.1, "Resolution at 1% FDR-FSC: {:.2f} Angstroem".format(resolution), fontsize=12);

	plt.xlabel("1/resolution [1/A]");
	plt.ylabel("FSC");
	plt.legend();

	plt.savefig('FSC.pdf', dpi=400);
	plt.close();

	#save txt file
	np.savetxt("FSC.txt", (resolutions, FSC));

#-------------------------------------------------------
def roundMapToVectorElements(map, apix): 
	
	sizeMap = map.shape;
	res = np.fft.rfftfreq(sizeMap[0], 1.0);
	res = res/float(apix);

	numRes = res.size;
	resSpacing = (res[1] - res[0])/2.0;
	
	for i in range(numRes):
	
		#get lower and upper bounds
		upper = 1.0/(res[i] - resSpacing);
		lower = 1.0/(res[i] + resSpacing);

		if i==0 :
			map[map>lower] = 100;
		
		elif i==(numRes-1):
			map[map<=upper] = 1.0/(res[i]);
		
		else: 
			map[(map>lower) & (map<=upper)] = 1.0/(res[i]);
		
	return map

#-------------------------------------------------------
def getNumAsymUnits(symmetry):

	#get symmetry group
	symGroup = symmetry[0];

	if symGroup == 'C':

		# get symmetry order
		symOrder = symmetry[1:];
		numAsymUnits = int(symOrder);

	elif symGroup == 'D':

		# get symmetry order
		symOrder = symmetry[1:];
		numAsymUnits = int(symOrder)*2;

	elif symGroup == 'O':

		numAsymUnits = 24;

	elif symGroup == 'I':

		numAsymUnits = 60;

	else:

		print("Symmetry not known. Exit ...");
		sys.exit();

	return numAsymUnits;


