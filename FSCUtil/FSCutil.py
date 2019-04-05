import numpy as np
from confidenceMapUtil import FDRutil 
import matplotlib.pyplot as plt
import math
from scipy.interpolate import RegularGridInterpolator

#-----------------------------------------------------
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
		freqj = np.fft.fftfreq(sizeMap[1], 1.0);

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

#---------------------------------------------------------------------------------
def makeCircularMask(map, sphereRadius):

	#***********************************************************
	#************** calculate spherical mask-map ***************
	#*** with sizei of the given map and radius sphereRadius ***
	#***********************************************************

	#some initialization
	mapSize = map.shape;

	x = np.linspace(-math.floor(mapSize[0]/2.0), -math.floor(mapSize[0]/2.0) + mapSize[0], mapSize[0]);
	y = np.linspace(-math.floor(mapSize[1]/2.0), -math.floor(mapSize[1]/2.0) + mapSize[1], mapSize[1]);
	z = np.linspace(-math.floor(mapSize[2]/2.0), -math.floor(mapSize[2]/2.0) + mapSize[2], mapSize[2]);

	xx, yy, zz = np.meshgrid(x, y, z, indexing='ij');

	radiusMap = np.sqrt(xx**2 + yy**2 + zz**2);

	#now extend mask with some smooth fall off
	gaussFallOffSigma = 2;

	tmpRadiusMap = radiusMap-sphereRadius;
	fallOffMap = np.exp(-((tmpRadiusMap)**2)/(2.0*gaussFallOffSigma**2));
	fallOffMap[fallOffMap < 0.000001] = 0.0;

	mask = fallOffMap;
	mask[tmpRadiusMap<0.0] = 1.0;

	return mask;

#---------------------------------------------------------------------------------
def makeHannWindow(map):

	#***********************************************************
	#*** generate Hann window with the size of the given map ***
	#***********************************************************

	#some initialization
	mapSize = map.shape;

	x = np.linspace(-math.floor(mapSize[0]/2.0), -math.floor(mapSize[0]/2.0) + mapSize[0], mapSize[0]);
	y = np.linspace(-math.floor(mapSize[1]/2.0), -math.floor(mapSize[1]/2.0) + mapSize[1], mapSize[1]);
	z = np.linspace(-math.floor(mapSize[2]/2.0), -math.floor(mapSize[2]/2.0) + mapSize[2], mapSize[2]);

	xx, yy, zz = np.meshgrid(x, y, z, indexing='ij');

	radiusMap = np.sqrt(xx**2 + yy**2 + zz**2);

	windowMap = 0.5*(1.0 - np.cos((2.0*np.pi*radiusMap/map.shape[0]) + np.pi));

	windowMap[radiusMap>(mapSize[0]/2.0)] = 0.0;

	return windowMap;

#--------------------------------------------------------
def FSC(halfMap1, halfMap2, maskData, apix, cutoff, numAsymUnits, localRes, verbose, permutedCorCoeffs):

	#********************************************
	#***** function that calculates the FSC *****
	#********************************************

	if localRes:
		maskCoeff = 0.23;
	else:
		maskCoeff = 0.7;

	if maskData is not None:
		halfMap1 = halfMap1*maskData;
		halfMap2 = halfMap2*maskData;

	#calculate frequency map
	freqMap = calculate_frequency_map(halfMap1);
	freqMap = freqMap/float(apix);

	#fourier transform the maps
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

		if (permutedCorCoeffs is not None):
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

	if localRes:
		if FSC[0] < 0.9:
			pVals[0] = 1.0;
		else:
			pVals[0] = 0.0;
		
		if FSC[1] < 0.9:
			pVals[1] = 1.0;
		else:
			pVals[1] = 0.0;

	# do FDR control of p-Values
	qVals_FDR = FDRutil.pAdjust(pVals, 'BH');
	qVals_FWER = FDRutil.pAdjust(pVals, 'Holm')

	tmpFSC = np.copy(FSC);
	tmpFSC[tmpFSC > cutoff] = 1.0;
	tmpFSC[tmpFSC <= cutoff] = 0.0;
	tmpFSC = 1.0 - tmpFSC;
	tmpFSC[0] = 0.0;	

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
	threshQValsFDR = threshQVals;

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

	threshQVals = np.copy(qVals_FWER);
	threshQVals[threshQVals <= 0.01] = 0.0; #signal
	threshQVals[threshQVals > 0.01] = 1.0 #no signal

	try:
		resolution_FWER = np.min(np.argwhere(threshQVals)) - 1;

		if resolution_FWER < 0:
			resolution_FWER = 0.0;
		else:
			if res[int(resolution_FWER)] == 0.0:
				resolution_FWER = 0.0;
			else:
				tmpFreq = res[int(resolution_FWER)] #+ (res[resolution_FDR + 1] - res[resolution_FDR]) / 2.0;
				resolution_FWER = float(1.0/tmpFreq);
	except:
		resolution_FWER = 2.0*apix;


	if verbose:
		print('Resolution at ' + repr(cutoff) + ' FSC threshold: ' + repr(round(resolution, 2)));
		print('Resolution at 1 % FDR: ' + repr(round(resolution_FDR, 2)) + ' Angstrom');
		print('Resolution at 1 % FWER: ' + repr(round(resolution_FWER, 2)) + ' Angstrom');

	return res, FSC, percentCutoffs, qVals_FWER, qVals_FDR, resolution_FDR, tmpPermutedCorCoeffs;

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
	permutedCorCoeffs = np.zeros(numPermutations);
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
	
	prevPValue = 0.0;
	for i in range(numPermutations):

		permutedSample2 = np.random.permutation(tmpSample2);
		#tmpCorCoeff = (np.sum((tmpSample1 * np.conj(permutedSample2)) + (tmpSample1ComplexConj * permutedSample2)));
		
		summand = tmpSample1ComplexConj * permutedSample2;
		tmpCorCoeff = np.sum(np.conj(summand) + summand);

		permutedCorCoeffs[i] = np.real(tmpCorCoeff);

		pValueCheckInterval = 100;
		if (i%pValueCheckInterval == 0) & (i>0):
			#calculate temporary p-value
			tmpPermutedCorCoeffs = np.real(permutedCorCoeffs[(i-pValueCheckInterval):i])/np.real(tmpFSCdenominator);		
		
			#calculate the pValue
			extremeCorCoeff = tmpPermutedCorCoeffs[tmpPermutedCorCoeffs>trueFSC];
			numExtreme = extremeCorCoeff.shape[0];
			newPValue = numExtreme/float(pValueCheckInterval);
			
			#if p-value accuracy is high enough, stop permutations
			if (newPValue - prevPValue) < 0.01:
				permutedCorCoeffs = permutedCorCoeffs[:(i+1)];
				numPermutations = i + 1;
				break;	

			prevPValue = newPValue;		
		

	permutedCorCoeffs = np.real(permutedCorCoeffs)/np.real(tmpFSCdenominator);

	"""#permutation sample
	start = time.time();	
	permutationsSample2 = np.tile(tmpSample2, (numPermutations, 1)).T;
	permutationsSample2 = permute_columns(permutationsSample2);
	summand1 = tmpSample1 * np.conj(permutationsSample2.T);
	summand2 = np.conj(tmpSample1) * permutationsSample2.T;
	permutationsSample2 = 0;
	tmpCorCoeff = np.real(np.sum(summand1 + summand2, 1));
	summand1 = 0.0;
	summand2 = 0.0
	permutedCorCoeffs = tmpCorCoeff/tmpFSCdenominator;
	end = time.time();
	print(end-start);
	"""
	
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

	#if sample size of FSC values is low, use a 0.9 FSC threshold	
	if maxSamples < 7.0:
		if trueFSC < 0.9:
			pValue = 1.0;
		else:
			pValue = 0.0;
		percentCutoffs = np.ones(percentCutoffs.shape);

	return pValue, percentCutoffs, threeSigma, threeSigmaCorr, permutedCorCoeffs;

#--------------------------------------------------------
def writeFSC(resolutions, FSC, percentCutoffs, qValuesFWER, qValuesFDR):

	#*******************************
	#******* write FSC plots *******
	#*******************************

	plt.plot(resolutions, FSC, label="FSC", linewidth=1.5);

	#plot percent cutoffs
	#for i in range(percentCutoffs.shape[1]):
		#plt.plot(resolutions[0:], percentCutoffs[0:, i], linewidth = 0.5, color='g');

	#threshold the adjusted pValues
	qValuesFDR[qValuesFDR<=0.01] = 0.0;
	qValuesFWER[qValuesFWER<=0.01] = 0.0;

	plt.plot(resolutions[0:][qValuesFDR==0.0], qValuesFDR[qValuesFDR==0.0]-0.05, 'xb', label="sign. at 1% FDR");
	plt.plot(resolutions[0:][qValuesFWER==0.0], qValuesFWER[qValuesFWER==0.0]-0.1, 'xm', label="sign. at 1% FWER");
	plt.axhline(0.5, linewidth = 0.5, color = 'r');
	plt.axhline(0.143, linewidth = 0.5, color = 'r');
	plt.axhline(0.0, linewidth = 0.5, color = 'b');
	
	plt.xlabel("1/resolution [1/A]");
	plt.ylabel("FSC");
	plt.legend();

	plt.savefig('FSC.png', dpi=300);
	plt.close();

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
def localResolutions(halfMap1, halfMap2, boxSize, stepSize, cutoff, apix, numAsymUnits, mask):

	#********************************************
	#****** calculate local resolutions by ******
	#********** local FSC-thresholding **********
	#********************************************

	print("Starting calculations of local resolutions ...");

	sizeMap = halfMap1.shape;
	localRes = np.zeros(sizeMap);
	resVector = np.fft.fftfreq(boxSize, apix);
	locRes = np.ones((len(range(boxSize, boxSize + sizeMap[0], stepSize)),len(range(boxSize, boxSize + sizeMap[1], stepSize)), len(range(boxSize, boxSize + sizeMap[2], stepSize))))*0.0;

	#pad the volumes
	paddedHalfMap1 = np.zeros((sizeMap[0] + 2*boxSize, sizeMap[1] + 2*boxSize, sizeMap[2] + 2*boxSize));
	paddedHalfMap2 = np.zeros((sizeMap[0] + 2*boxSize, sizeMap[1] + 2*boxSize, sizeMap[2] + 2*boxSize));
	paddedLocalRes = np.ones((sizeMap[0] + 2*boxSize, sizeMap[1] + 2*boxSize, sizeMap[2] + 2*boxSize));
	paddedMask = np.zeros((sizeMap[0] + 2*boxSize, sizeMap[1] + 2*boxSize, sizeMap[2] + 2*boxSize));

	paddedHalfMap1[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1], boxSize: boxSize + sizeMap[2]] = halfMap1;
	paddedHalfMap2[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1], boxSize: boxSize + sizeMap[2]] = halfMap2;
	paddedLocalRes[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1], boxSize: boxSize + sizeMap[2]] = localRes;
	paddedMask[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1], boxSize: boxSize + sizeMap[2]] = mask;

	halfBoxSize = int(boxSize/2.0);
	halfStepSize = int(stepSize/2.0);

	#make Hann window
	hannWindow = makeHannWindow(np.zeros((boxSize, boxSize, boxSize)));

	numCalculations = len(range(boxSize, boxSize + sizeMap[0], stepSize)) * len(range(boxSize, boxSize + sizeMap[1], stepSize)) * len(range(boxSize, boxSize + sizeMap[0], stepSize));
	print("Total number of calculations: " + repr(numCalculations));

	#****************************************************
	#******** get  initial permuted CorCoeffs ***********
	#****************************************************

	print("Do initial permuations ...");
	for i in range(10):

		xInd = np.random.randint(boxSize, sizeMap[0] + boxSize);
		yInd = np.random.randint(boxSize, sizeMap[1] + boxSize);
		zInd = np.random.randint(boxSize, sizeMap[2] + boxSize);

		while( paddedMask[xInd, yInd, zInd] < 0.1 ):

			xInd = np.random.randint(boxSize, sizeMap[0] + boxSize);
			yInd = np.random.randint(boxSize, sizeMap[1] + boxSize);
			zInd = np.random.randint(boxSize, sizeMap[2] + boxSize);

		window_halfmap1 = paddedHalfMap1[xInd - halfBoxSize: xInd - halfBoxSize + boxSize,
					  yInd - halfBoxSize: yInd - halfBoxSize + boxSize, zInd - halfBoxSize: zInd - halfBoxSize + boxSize];
		window_halfmap2 = paddedHalfMap2[xInd - halfBoxSize: xInd - halfBoxSize + boxSize,
					  yInd - halfBoxSize: yInd - halfBoxSize + boxSize, zInd - halfBoxSize: zInd - halfBoxSize + boxSize];

		# apply hann window
		window_halfmap1 = window_halfmap1 * hannWindow;
		window_halfmap2 = window_halfmap2 * hannWindow;

		res, _, _, _, _, _ , tmpPermutedCorCoeffs = FSC(window_halfmap1, window_halfmap2, None, apix, cutoff, numAsymUnits, True, False, None);

		if i == 0:
			#initialize the array of correlation coefficients
			permutedCorCoeffs = tmpPermutedCorCoeffs;
		else:
			#append the correlation coefficients
			for resInd in range(len(tmpPermutedCorCoeffs)):
				permutedCorCoeffs[resInd] = np.append(permutedCorCoeffs[resInd], tmpPermutedCorCoeffs[resInd]);
	

	#****************************************************
	#********* calculate the local resolutions **********
	#****************************************************

	print("Do local FSC calculations ...");
	calcInd = 0;
	iInd = 0;
	jInd = 0;
	kInd = 0;
	for i in range(boxSize, boxSize + sizeMap[0], stepSize):
		jInd = 0;
		for j in range(boxSize, boxSize + sizeMap[1], stepSize):
			kInd = 0;
			for k in range(boxSize, boxSize + sizeMap[2], stepSize):

				if paddedMask[i, j, k] > 0.99:
					window_halfmap1 = paddedHalfMap1[i - halfBoxSize: i - halfBoxSize + boxSize, j - halfBoxSize: j - halfBoxSize + boxSize, k - halfBoxSize: k - halfBoxSize + boxSize];
					window_halfmap2 = paddedHalfMap2[i - halfBoxSize: i - halfBoxSize + boxSize, j - halfBoxSize: j - halfBoxSize + boxSize, k - halfBoxSize: k - halfBoxSize + boxSize];

					#apply hann window
					window_halfmap1 = window_halfmap1 * hannWindow;
					window_halfmap2 = window_halfmap2 * hannWindow;

					_, _ , _, _, _, tmpRes, _ = FSC(window_halfmap1, window_halfmap2, None, apix, cutoff, numAsymUnits, True, False, permutedCorCoeffs);

					paddedLocalRes[i - halfStepSize: i - halfStepSize + stepSize, j - halfStepSize: j - halfStepSize + stepSize, k - halfStepSize: k - halfStepSize + stepSize] = tmpRes;
					#paddedLocalRes[i - halfBoxSize: i - halfBoxSize + boxSize,j - halfBoxSize: j - halfBoxSize + boxSize, k - halfBoxSize: k - halfBoxSize + boxSize] = tmpRes;
					locRes[iInd, jInd, kInd] = tmpRes;
				else:
					paddedLocalRes[i - halfStepSize: i - halfStepSize + stepSize, j - halfStepSize: j - halfStepSize + stepSize, k - halfStepSize: k - halfStepSize + stepSize] = 0.0;
					locRes[iInd, jInd, kInd] = 0.0;

				calcInd = calcInd + 1;
				kInd = kInd + 1;
				
				#print output
				progress = calcInd/float(numCalculations);
				if calcInd%(int(numCalculations/20.0)) == 0:
					output = "%.1f" %(progress*100) + "% finished ..." ;
					print(output);

			jInd = jInd + 1;
		iInd = iInd + 1;

	#*************************************
	#********** do interpolation *********
	#*************************************

	x = np.linspace(1, 10, locRes.shape[0]);
	y = np.linspace(1, 10, locRes.shape[1]);
	z = np.linspace(1, 10, locRes.shape[2]);

	my_interpolating_function = RegularGridInterpolator((x, y, z), locRes, method='linear')

	xNew = np.linspace(1, 10, sizeMap[0]);
	yNew = np.linspace(1, 10, sizeMap[1]);
	zNew = np.linspace(1, 10, sizeMap[2]);

	xInd, yInd, zInd = np.meshgrid(xNew, yNew, zNew, indexing='ij', sparse=True);

	localRes = my_interpolating_function((xInd, yInd, zInd));

	localRes[mask <= 0.99] = 0.0;

	return localRes;
