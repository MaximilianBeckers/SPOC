# Author: Maximilian Beckers, EMBL Heidelberg, Sachse Group (2019)

# import some stuff
from FSCUtil import FSCutil, localResolutions
from confidenceMapUtil import FDRutil
import numpy as np
import argparse, os, sys
import os.path
import time
import math
import mrcfile

# *************************************************************
# ****************** Commandline input ************************
# *************************************************************

cmdl_parser = argparse.ArgumentParser(
	prog=sys.argv[0],
	description='*** Thresholding of FSC curves by FDR control ***',
	formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-halfmap1', '--halfmap1', metavar="halfmap1.mrc", type=str, required=True,
						 help='Input filename halfmap 1');
cmdl_parser.add_argument('-halfmap2', '--halfmap2', metavar="halfmap2.mrc", type=str, required=True,
						 help='Input filename halfmap 2');
cmdl_parser.add_argument('-p', '--apix', metavar="apix", type=float, required=False,
						 help='pixel Size of input map');
cmdl_parser.add_argument('-localResolutions', action='store_true', default=False,
						 help='Flag for calculation of local resolution');
cmdl_parser.add_argument('-w', '--window_size', metavar="windowSize", type=float, required=False,
						 help="Input window size for local Amplitude scaling and background noise estimation");
cmdl_parser.add_argument('-stepSize', '--stepSize', metavar="stepSize_locScale", type=int, required=False,
						 help="Voxels to skip for local amplitude scaling");
cmdl_parser.add_argument('-symmetry', '--symmetry', type=str, required=False,
						 help="symmetry for correction of symmetry effects")
cmdl_parser.add_argument('-numAsymUnits', '--numAsymUnits', type=int, required=False,
						 help="number of asymmtetric units for correction of symmetry effects")
cmdl_parser.add_argument('-bFactor', '--bFactor', type=float, required=False,
						 help="B-Factor for sharpening of the map")

# ************************************************************
# ********************** main function ***********************
# ************************************************************

def main():
	start = time.time();

	print('***************************************************');
	print('******* Significance analysis of FSC curves *******');
	print('***************************************************');

	# get command line input
	args = cmdl_parser.parse_args();

	#read the half maps
	halfMap1 = mrcfile.open(args.halfmap1, mode='r');
	halfMap2 = mrcfile.open(args.halfmap2, mode='r');

	halfMap1Data = np.copy(halfMap1.data);
	halfMap2Data = np.copy(halfMap2.data);

	#get size of map
	sizeMap = halfMap2Data.shape;

	#set pixel size
	apix = float(halfMap1.voxel_size.x);
	if args.apix is not None:
		print('Pixel size set to {:.3f} Angstroem. (Pixel size encoded in map: {:.3f})'.format(args.apix, apix));
		apix = args.apix;
	else:
		print('Pixel size was read as {:.3f} Angstroem. If this is incorrect, please specify with -p pixelSize'.format(apix));
		args.apix = apix;

	# set output filename
	splitFilename = os.path.splitext(os.path.basename(args.halfmap1));
	outputFilename_LocRes = splitFilename[0] + "_localResolutions.mrc";
	outputFilename_PostProcessed = splitFilename[0] + "_postProcessed.mrc";

	#handle window size for local FSC
	if args.window_size is not None:
		wn = args.window_size;
		wn = int(wn);
	else:
		wn = 20;

	#handle step size for local FSC
	if args.stepSize is None:
		stepSize = float(sizeMap[0]*sizeMap[1]*sizeMap[2])/300000.0;
		stepSize = max(int(math.ceil(stepSize**(1.0/3.0))),1);
	else:
		stepSize = int(args.stepSize);

	if not args.localResolutions:
		if args.numAsymUnits is not None:

			numAsymUnits = args.numAsymUnits;
			print('Using user provided number of asymmetric units, given as {:d}'.format(numAsymUnits));
		else:
			if args.symmetry is not None:

				numAsymUnits = FSCutil.getNumAsymUnits(args.symmetry);
				print('Using provided ' + args.symmetry + ' symmetry. Number of asymmetric units: {:d}'.format(numAsymUnits));
			else:
				numAsymUnits = 1;
				print('Using C1 symmetry. Number of asymmetric units: {:d}'.format(numAsymUnits));
	else:
		#if local resolutions are calculated, no symmetry correction needed
		print( "Using a step size of {:d} voxel. If you prefer another one, please specify with -step.".format(stepSize));
		print('Calculating local resolutions. No symmetry correction necessary.');
		numAsymUnits = 1.0;

	#make the mask
	print("Using a circular mask ...");
	maskData = FSCutil.makeCircularMask(halfMap1Data, (np.min(halfMap1Data.shape) / 2.0) - 4.0);
	maskBFactor = FSCutil.makeCircularMask(halfMap1Data, (np.min(halfMap1Data.shape) / 4.0) - 4.0);

	#*******************************************
	#********** no local Resolutions ***********
	#*******************************************

	if not args.localResolutions:
		res, FSC, percentCutoffs, pValues, qValsFDR, resolution, _ = FSCutil.FSC(halfMap1Data, halfMap2Data,
																					  maskData, apix, 0.143,
																					  numAsymUnits, False, True, None);
		# write the FSC
		FSCutil.writeFSC(res, FSC, qValsFDR, pValues);
		
		if resolution < 8.0:
			#estimate b-factor and sharpen the map
			bFactor = FSCutil.estimateBfactor(0.5*(halfMap1Data+halfMap2Data), resolution, apix, maskBFactor);

			if args.bFactor is not None:
				bFactor = args.bFactor;
				print('Using a user-specified B-factor of {:.2f} for map sharpening'.format(-bFactor));
			else:
				print('Using a B-factor of {:.2f} for map sharpening.'.format(-bFactor));

			processedMap = FDRutil.sharpenMap(0.5*(halfMap1Data+halfMap2Data), -bFactor, apix, resolution);

			#write the post-processed map
			postProcMRC = mrcfile.new(outputFilename_PostProcessed, overwrite=True);
			postProc= np.float32(processedMap);
			postProcMRC.set_data(postProc);
			postProcMRC.voxel_size = apix;
			postProcMRC.close();
			
			output = "Saved sharpened and filtered map to: " + outputFilename_PostProcessed;
			print(output);

	#*******************************************
	#********* calc local Resolutions **********
	#*******************************************
	else:
		FSCcutoff = 0.5;
		localResMap = localResolutions.localResolutions(halfMap1Data, halfMap2Data, wn, stepSize, FSCcutoff, apix, numAsymUnits,
											   maskData);

		#write the local resolution map
		localResMapMRC = mrcfile.new(outputFilename_LocRes, overwrite=True);
		localResMap = np.float32(localResMap);
		localResMapMRC.set_data(localResMap);
		localResMapMRC.voxel_size = apix;
		localResMapMRC.close();

		output = "Saved local resolutions map to: " + outputFilename_LocRes;
		print(output);

	end = time.time();
	totalRuntime = end - start;
	
	print("****** Summary ******");
	print("Runtime: %.2f" %totalRuntime);

if (__name__ == "__main__"):
	main()
