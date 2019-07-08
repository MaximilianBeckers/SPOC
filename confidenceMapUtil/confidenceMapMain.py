import numpy as np
import subprocess
import math
import gc
import os
import sys
from confidenceMapUtil import mapUtil, locscaleUtil, FDRutil

#--------------------------------------------------------------------------
def calculateConfidenceMap(em_map, apix, noiseBox, testProc, ecdf, lowPassFilter_resolution, method, window_size, locResMap,
						   meanMap, varMap, fdr, modelMap, stepSize, windowSizeLocScale, mpi):

	#*********************************************
	#******* this function calc. confMaps ********
	#*********************************************

	# get boxCoordinates
	if noiseBox is None:
		boxCoord = 0;
	else:
		boxCoord = noiseBox;

	# set test procdure
	if testProc is not None:
		testProc = testProc;
	else:
		testProc = 'rightSided';

	# set ECDF
	if ecdf:
		ECDF = 1;
	else:
		ECDF = 0;

	sizeMap = em_map.shape;

	if lowPassFilter_resolution is not None:
		frequencyMap = FDRutil.calculate_frequency_map(em_map);
		providedRes = apix/float(lowPassFilter_resolution);
		em_map = FDRutil.lowPassFilter(np.fft.rfftn(em_map), frequencyMap, providedRes, em_map.shape);

	# handle FDR correction procedure
	if method is not None:
		method = method;
	else:
		# default is Benjamini-Yekutieli
		method = 'BY';

	if window_size is not None:
		wn = window_size;
		wn = int(wn);
		if wn < 20:
			print("Provided window size is quite small. Please think about potential inaccuracies of your noise estimates!");
	else:
		wn = max(int(0.05 * sizeMap[0]), 10);

	if windowSizeLocScale is not None:
		wn_locscale = windowSizeLocScale;
		if window_size is None:
			wn = int(wn_locscale);
	else:
		wn_locscale = None;

	if stepSize is None:
		stepSize = 5;

	# generate a circular Mask
	sphere_radius = (np.max(sizeMap) // 2);
	circularMaskData = mapUtil.makeCircularMask(np.copy(em_map), sphere_radius);

	# plot locations of noise estimation
	if modelMap is None:
		pp = mapUtil.makeDiagnosticPlot(em_map, wn, False, boxCoord);
		pp.savefig("diag_image.pdf");
		pp.close();
	else:
		pp = mapUtil.makeDiagnosticPlot(em_map, wn, True, boxCoord);
		pp.savefig("diag_image.pdf");
		pp.close();


	# estimate noise statistics
	if ((locResMap is None) & (modelMap is None)):  # if no local Resolution map is given,don't do any filtration

		FDRutil.checkNormality(em_map, wn, boxCoord);
		mean, var, _ = FDRutil.estimateNoiseFromMap(em_map, wn, boxCoord);

		if varMap is not None:
			var = varMap;
		if meanMap is not None:
			mean = meanMap;

		if np.isscalar(mean) and np.isscalar(var):
			output = "Estimated noise statistics: mean: " + repr(mean) + " and variance: " + repr(var);
		else:
			output = "Using user provided noise statistics";
			print(output);

		locFiltMap = None;
		locScaleMap = None;

	elif (locResMap is not None) & (modelMap is None):  # do localFiltration and estimate statistics from this map

		FDRutil.checkNormality(em_map, wn, boxCoord);
		em_map, mean, var, ECDF = mapUtil.localFiltration(em_map, locResMap, apix, True, wn, boxCoord, ECDF);
		#locFiltMap = FDRutil.studentizeMap(em_map, mean, var);
		locFiltMap = em_map;
		locScaleMap = None;
	else:
		em_map, mean, var, ECDF = locscaleUtil.launch_amplitude_scaling(em_map, modelMap, apix, stepSize, wn_locscale, wn, method, locResMap, boxCoord, mpi, ECDF );
		#locScaleMap = FDRutil.studentizeMap(em_map, mean, var);
		locScaleMap = em_map;
		locFiltMap = None;

	# calculate the qMap
	if method == 'BH':
		qMap = FDRutil.calcQMap(em_map, mean, var, ECDF, wn, boxCoord, circularMaskData, 'BH', testProc);
		error = 'FDR';
	elif method == 'Hochberg':
		qMap = FDRutil.calcQMap(em_map, mean, var, ECDF, wn, boxCoord, circularMaskData, 'Hochberg', testProc);
		error = 'FWER';
	elif method == 'Holm':
		qMap = FDRutil.calcQMap(em_map, mean, var, ECDF, wn, boxCoord, circularMaskData, 'Holm', testProc);
		error = 'FWER';
	else:
		qMap = FDRutil.calcQMap(em_map, mean, var, ECDF, wn, boxCoord, circularMaskData, 'BY', testProc);
		error = 'FDR';

	#if local processing wished, write that out
	if locFiltMap is not None:
		em_map = locFiltMap;
	if locScaleMap is not None:
		em_map = locScaleMap;

	if ((locResMap is None) & (modelMap is None)):
		# threshold the qMap
		binMap1 = FDRutil.binarizeMap(qMap, 0.01);
		binMap001 = FDRutil.binarizeMap(qMap, 0.0001);

		# apply the thresholded qMapFDR to data
		maskedMap1 = np.multiply(binMap1, np.copy(em_map));
		minMapValue = np.min(maskedMap1[np.nonzero(maskedMap1)]);
		output = "Calculated map threshold: %.3f" %minMapValue + " at a " + error + " of " + repr(1) + "%.";
		print(output);

		# apply the thresholded qMapFWER to data
		maskedMap001 = np.multiply(binMap001, np.copy(em_map));
		minMapValue = np.min(maskedMap001[np.nonzero(maskedMap001)]);
		output = "Calculated map threshold: %.3f" %minMapValue + " at a " + error + " of " + repr(0.01) + "%.";
		print(output);

	# invert qMap for visualization tools
	confidenceMap = np.subtract(1.0, qMap);

	# apply lowpass-filtered mask to maps
	confidenceMap = np.multiply(confidenceMap, circularMaskData);


	return confidenceMap, locFiltMap, locScaleMap, mean, var;
