import numpy as np
import functools
import multiprocessing
import math
from FSCUtil import FSCutil
from confidenceMapUtil import FDRutil
from scipy.interpolate import RegularGridInterpolator

#------------------------------------------------------------
def localResolutions2D(halfMap1, halfMap2, boxSize, stepSize, cutoff, apix, numAsymUnits, mask, maskPermutation, lowRes):

	# ********************************************
	# ****** calculate local resolutions by ******
	# ********** local FSC-thresholding **********
	# ********************************************

	print("Starting calculations of local resolutions ...");

	sizeMap = halfMap1.shape;
	locRes = np.zeros((len(range(boxSize, boxSize + sizeMap[0], stepSize)),
					  len(range(boxSize, boxSize + sizeMap[1], stepSize))));

	# pad the volumes
	paddedHalfMap1 = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize));
	paddedHalfMap2 = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize));
	paddedMask = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize));
	paddedMaskPermutation = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize));

	paddedHalfMap1[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1]] = halfMap1;
	paddedHalfMap2[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1]] = halfMap2;
	paddedMask[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1]] = mask;
	paddedMaskPermutation[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1]] = maskPermutation;

	halfBoxSize = int(boxSize / 2.0);

	# make Hann window
	hannWindow = FDRutil.makeHannWindow(np.zeros((boxSize, boxSize)));

	numCalculations = len(range(boxSize, boxSize + sizeMap[0], stepSize)) * len(
		range(boxSize, boxSize + sizeMap[1], stepSize));
	print("Total number of calculations: " + repr(numCalculations));

	# ****************************************************
	# ********* get initial permuted CorCoeffs ***********
	# ****************************************************

	print("Do initial permuations ...");
	for i in range(1):

		xInd = np.random.randint(boxSize, sizeMap[0] + boxSize);
		yInd = np.random.randint(boxSize, sizeMap[1] + boxSize);

		#xInd = np.random.randint(sizeMap[0]/2 - sizeMap[0]/8 + boxSize, sizeMap[0]/2 + sizeMap[0]/8 + boxSize);
		#yInd = np.random.randint(sizeMap[1]/2 - sizeMap[1]/8 + boxSize, sizeMap[1]/2 + sizeMap[1]/8 + boxSize);
		#zInd = np.random.randint(sizeMap[2]/2 - sizeMap[2]/8 + boxSize, sizeMap[2]/2 + sizeMap[2]/8 + boxSize);

		#generate new locations until one is found in the mask
		while ((paddedMaskPermutation[xInd, yInd] < 0.5)):

			xInd = np.random.randint(boxSize, sizeMap[0] + boxSize);
			yInd = np.random.randint(boxSize, sizeMap[1] + boxSize);

			#xInd = np.random.randint(sizeMap[0] / 2 - sizeMap[0] / 8 + boxSize,
			#						 sizeMap[0] / 2 + sizeMap[0] / 8 + boxSize);
			#yInd = np.random.randint(sizeMap[1] / 2 - sizeMap[1] / 8 + boxSize,
			#						 sizeMap[1] / 2 + sizeMap[1] / 8 + boxSize);
			#zInd = np.random.randint(sizeMap[2] / 2 - sizeMap[2] / 8 + boxSize,
			#						 sizeMap[2] / 2 + sizeMap[2] / 8 + boxSize);

		#get windowed parts
		windowHalfmap1 = paddedHalfMap1[xInd - halfBoxSize: xInd - halfBoxSize + boxSize,
						 yInd - halfBoxSize: yInd - halfBoxSize + boxSize];
		windowHalfmap2 = paddedHalfMap2[xInd - halfBoxSize: xInd - halfBoxSize + boxSize,
						 yInd - halfBoxSize: yInd - halfBoxSize + boxSize];

		# apply hann window
		windowHalfmap1 = windowHalfmap1 * hannWindow;
		windowHalfmap2 = windowHalfmap2 * hannWindow;

		res, _, _, _, _, _, tmpPermutedCorCoeffs = FSCutil.FSC(windowHalfmap1, windowHalfmap2, None, apix, cutoff, numAsymUnits,
													   False, False, None, True);

		if i == 0:
			# initialize the array of correlation coefficients
			permutedCorCoeffs = tmpPermutedCorCoeffs;
		else:
			# append the correlation coefficients
			for resInd in range(len(tmpPermutedCorCoeffs)):
				permutedCorCoeffs[resInd] = np.append(permutedCorCoeffs[resInd], tmpPermutedCorCoeffs[resInd]);


	# ****************************************************
	# ********* calculate the local resolutions **********
	# ****************************************************

	print("Do local FSC calculations ...");

	# generate partial function to loop over the whole map
	partialLoopOverMap = functools.partial(loopOverMap, paddedMask=paddedMask, paddedHalfMap1=paddedHalfMap1,
										 paddedHalfMap2=paddedHalfMap2, boxSize=boxSize, sizeMap=sizeMap,
										 stepSize=stepSize, halfBoxSize=halfBoxSize,
										 hannWindow=hannWindow, apix=apix, cutoff=cutoff, numAsymUnits=numAsymUnits,
										 permutedCorCoeffs=permutedCorCoeffs);

	#parallelized local resolutions
	numCores = multiprocessing.cpu_count();
	print("Using {:d} cores. This might take a few minutes ...".format(numCores));
	iIterable = range(boxSize, boxSize + sizeMap[0], stepSize);

	#initialize parallel processes
	lenInt = int(math.floor(len(iIterable)/float(numCores)));

	queue = multiprocessing.Queue();

	#start process for each core and run in parallel
	for i in range(numCores):

		#split the iterable
		startInd = (i*lenInt);
		endInd = (i+1)*lenInt;
		if i == (numCores-1):
			seq = range(iIterable[startInd], iIterable[len(iIterable)-1]+stepSize, stepSize);
		else:
			seq = range(iIterable[startInd], iIterable[endInd], stepSize);

		#start the respective process
		proc = multiprocessing.Process(target=partialLoopOverMap, args=(seq, queue,));
		proc.start();

	#addition of indiviual local resolution maps to produce the final one
	for i in range(numCores):
		locRes = locRes + queue.get();


	if lowRes is not None:  # if low-resolution bound is give, use it
			locRes[locRes > lowRes] = lowRes;

	# *************************************
	# ********** do interpolation *********
	# *************************************

	print("Interpolating local Resolutions ...");
	x = np.linspace(1, 10, locRes.shape[0]);
	y = np.linspace(1, 10, locRes.shape[1]);

	myInterpolatingFunction = RegularGridInterpolator((x, y), locRes, method='linear')

	xNew = np.linspace(1, 10, sizeMap[0]);
	yNew = np.linspace(1, 10, sizeMap[1]);

	xInd, yInd = np.meshgrid(xNew, yNew, indexing='ij', sparse=True);

	localRes = myInterpolatingFunction((xInd, yInd));

	localRes[mask <= 0.1] = 0.0;

	return localRes;

#-----------------------------------------------------------------
def loopOverMap(iSeq, queue,  paddedMask, paddedHalfMap1, paddedHalfMap2, boxSize, sizeMap,
				stepSize, halfBoxSize, hannWindow, apix, cutoff, numAsymUnits, permutedCorCoeffs):

	# ********************************************
	# ******* iterate over the map and calc ******
	# ************ local resolutions *************
	# ********************************************

	locRes = np.zeros((len(range(boxSize, boxSize + sizeMap[0], stepSize)),
					   len(range(boxSize, boxSize + sizeMap[1], stepSize))));

	for i in iSeq:

		iInd = int((i-boxSize)/stepSize);
		jInd = 0;

		for j in range(boxSize, boxSize + sizeMap[1], stepSize):

			if paddedMask[i, j] > 0.99:

				window_halfmap1 = paddedHalfMap1[i - halfBoxSize: i - halfBoxSize + boxSize,
									  j - halfBoxSize: j - halfBoxSize + boxSize];
				window_halfmap2 = paddedHalfMap2[i - halfBoxSize: i - halfBoxSize + boxSize,
									  j - halfBoxSize: j - halfBoxSize + boxSize];

				# apply hann window
				window_halfmap1 = window_halfmap1 * hannWindow;
				window_halfmap2 = window_halfmap2 * hannWindow;

				_, _, _, _, _, tmpRes, _ = FSCutil.FSC(window_halfmap1, window_halfmap2, None, apix, cutoff,
														   numAsymUnits,
														   False, False, None, True);

				locRes[iInd, jInd] = tmpRes;

			else:
				locRes[iInd, jInd] = 0.0;

			jInd = jInd + 1;

	#push back the local resolution map to the list
	queue.put(locRes);