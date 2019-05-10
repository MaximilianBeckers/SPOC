import numpy as np
from FSCUtil import FSCutil
from confidenceMapUtil import FDRutil
from scipy.interpolate import RegularGridInterpolator


def localResolutions(halfMap1, halfMap2, boxSize, stepSize, cutoff, apix, numAsymUnits, mask):
	# ********************************************
	# ****** calculate local resolutions by ******
	# ********** local FSC-thresholding **********
	# ********************************************

	print("Starting calculations of local resolutions ...");

	sizeMap = halfMap1.shape;
	localRes = np.zeros(sizeMap);
	resVector = np.fft.fftfreq(boxSize, apix);
	locRes = np.ones((len(range(boxSize, boxSize + sizeMap[0], stepSize)),
					  len(range(boxSize, boxSize + sizeMap[1], stepSize)),
					  len(range(boxSize, boxSize + sizeMap[2], stepSize)))) * 0.0;

	# pad the volumes
	paddedHalfMap1 = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize, sizeMap[2] + 2 * boxSize));
	paddedHalfMap2 = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize, sizeMap[2] + 2 * boxSize));
	paddedLocalRes = np.ones((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize, sizeMap[2] + 2 * boxSize));
	paddedMask = np.zeros((sizeMap[0] + 2 * boxSize, sizeMap[1] + 2 * boxSize, sizeMap[2] + 2 * boxSize));

	paddedHalfMap1[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1],
	boxSize: boxSize + sizeMap[2]] = halfMap1;
	paddedHalfMap2[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1],
	boxSize: boxSize + sizeMap[2]] = halfMap2;
	paddedLocalRes[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1],
	boxSize: boxSize + sizeMap[2]] = localRes;
	paddedMask[boxSize: boxSize + sizeMap[0], boxSize: boxSize + sizeMap[1], boxSize: boxSize + sizeMap[2]] = mask;

	halfBoxSize = int(boxSize / 2.0);
	halfStepSize = int(stepSize / 2.0);

	# make Hann window
	hannWindow = FDRutil.makeHannWindow(np.zeros((boxSize, boxSize, boxSize)));

	numCalculations = len(range(boxSize, boxSize + sizeMap[0], stepSize)) * len(
		range(boxSize, boxSize + sizeMap[1], stepSize)) * len(range(boxSize, boxSize + sizeMap[0], stepSize));
	print("Total number of calculations: " + repr(numCalculations));

	# ****************************************************
	# ******** get  initial permuted CorCoeffs ***********
	# ****************************************************

	print("Do initial permuations ...");
	for i in range(10):

		xInd = np.random.randint(boxSize, sizeMap[0] + boxSize);
		yInd = np.random.randint(boxSize, sizeMap[1] + boxSize);
		zInd = np.random.randint(boxSize, sizeMap[2] + boxSize);

		while (paddedMask[xInd, yInd, zInd] < 0.1):
			xInd = np.random.randint(boxSize, sizeMap[0] + boxSize);
			yInd = np.random.randint(boxSize, sizeMap[1] + boxSize);
			zInd = np.random.randint(boxSize, sizeMap[2] + boxSize);

		windowHalfmap1 = paddedHalfMap1[xInd - halfBoxSize: xInd - halfBoxSize + boxSize,
						 yInd - halfBoxSize: yInd - halfBoxSize + boxSize,
						 zInd - halfBoxSize: zInd - halfBoxSize + boxSize];
		windowHalfmap2 = paddedHalfMap2[xInd - halfBoxSize: xInd - halfBoxSize + boxSize,
						 yInd - halfBoxSize: yInd - halfBoxSize + boxSize,
						 zInd - halfBoxSize: zInd - halfBoxSize + boxSize];

		# apply hann window
		windowHalfmap1 = windowHalfmap1 * hannWindow;
		windowHalfmap2 = windowHalfmap2 * hannWindow;

		res, _, _, _, _, _, tmpPermutedCorCoeffs = FSCutil.FSC(windowHalfmap1, windowHalfmap2, None, apix, cutoff, numAsymUnits,
													   True, False, None);

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
					window_halfmap1 = paddedHalfMap1[i - halfBoxSize: i - halfBoxSize + boxSize,
									  j - halfBoxSize: j - halfBoxSize + boxSize,
									  k - halfBoxSize: k - halfBoxSize + boxSize];
					window_halfmap2 = paddedHalfMap2[i - halfBoxSize: i - halfBoxSize + boxSize,
									  j - halfBoxSize: j - halfBoxSize + boxSize,
									  k - halfBoxSize: k - halfBoxSize + boxSize];

					# apply hann window
					window_halfmap1 = window_halfmap1 * hannWindow;
					window_halfmap2 = window_halfmap2 * hannWindow;

					_, _, _, _, _, tmpRes, _ = FSCutil.FSC(window_halfmap1, window_halfmap2, None, apix, cutoff, numAsymUnits,
												   True, False, permutedCorCoeffs);

					paddedLocalRes[i - halfStepSize: i - halfStepSize + stepSize,
					j - halfStepSize: j - halfStepSize + stepSize,
					k - halfStepSize: k - halfStepSize + stepSize] = tmpRes;
					# paddedLocalRes[i - halfBoxSize: i - halfBoxSize + boxSize,j - halfBoxSize: j - halfBoxSize + boxSize, k - halfBoxSize: k - halfBoxSize + boxSize] = tmpRes;
					locRes[iInd, jInd, kInd] = tmpRes;
				else:
					paddedLocalRes[i - halfStepSize: i - halfStepSize + stepSize,
					j - halfStepSize: j - halfStepSize + stepSize, k - halfStepSize: k - halfStepSize + stepSize] = 0.0;
					locRes[iInd, jInd, kInd] = 0.0;

				calcInd = calcInd + 1;
				kInd = kInd + 1;

				# print output
				progress = calcInd / float(numCalculations);
				if calcInd % (int(numCalculations / 20.0)) == 0:
					output = "%.1f" % (progress * 100) + "% finished ...";
					print(output);

			jInd = jInd + 1;
		iInd = iInd + 1;

	# *************************************
	# ********** do interpolation *********
	# *************************************

	x = np.linspace(1, 10, locRes.shape[0]);
	y = np.linspace(1, 10, locRes.shape[1]);
	z = np.linspace(1, 10, locRes.shape[2]);

	myInterpolatingFunction = RegularGridInterpolator((x, y, z), locRes, method='linear')

	xNew = np.linspace(1, 10, sizeMap[0]);
	yNew = np.linspace(1, 10, sizeMap[1]);
	zNew = np.linspace(1, 10, sizeMap[2]);

	xInd, yInd, zInd = np.meshgrid(xNew, yNew, zNew, indexing='ij', sparse=True);

	localRes = myInterpolatingFunction((xInd, yInd, zInd));

	localRes[mask <= 0.99] = 0.0;

	return localRes;