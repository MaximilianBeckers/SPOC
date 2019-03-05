# Author: Maximilian Beckers, EMBL Heidelberg, Sachse Group (2019)

# import some stuff
from FSCUtil import FSCutil
import numpy as np
import argparse, os, sys
import os.path
import time
import mrcfile

# *************************************************************
# ****************** Commandline input ************************
# *************************************************************

cmdl_parser = argparse.ArgumentParser(
	prog=sys.argv[0],
	description='*** Thresholding of FSC curves by FDR control ***',
	formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-halfmap1', '--halfmap1', metavar="halfmap1.mrc", type=str, required=True,
						 help='Input filename  halfmap 1');
cmdl_parser.add_argument('-halfmap2', '--halfmap2', metavar="halfmap2.mrc", type=str, required=True,
						 help='Input filename halfmap 2');
cmdl_parser.add_argument('-p', '--apix', metavar="apix", type=float, required=False,
						 help='pixel Size of input map');
cmdl_parser.add_argument('-fdr', '--fdr', metavar="fdr", type=float, required=False,
						 help='False Discovery Rate');
cmdl_parser.add_argument('-localResolutions', action='store_true', default=False,
						 help='Flag for calculation of local resolution');
cmdl_parser.add_argument('-method', metavar="method", type=str, required=False,
						 help="Method for multiple testing correction. 'BY' for Benjamini-Yekutieli, 'BH' for Benjamini-Hochberg or 'Holm' for Holm FWER control");
cmdl_parser.add_argument('-w', '--window_size', metavar="windowSize", type=float, required=False,
						 help="Input window size for local Amplitude scaling and background noise estimation");
cmdl_parser.add_argument('-o', '--outputFilename', metavar="output.mrc", type=str, required=False,
						 help="Name of the output for local resolutions map. Example: locRes.mrc");
cmdl_parser.add_argument('-testProc', '--testProc', type=str, required=False,
						 help="choose between right, left and two-sided testing");
cmdl_parser.add_argument('-stepSize', '--stepSize', metavar="stepSize_locScale", type=int, required=False,
						 help="Voxels to skip for local amplitude scaling");
cmdl_parser.add_argument('-numAsymUnits', '--numAsymUnits', type=int, required=False,
						 help="number of asymmtetric units for correction of symmetry effects")
cmdl_parser.add_argument('-mask', metavar="mask.mrc", type=str, required=False,
						 help='Input filename mask')

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


	#set pixel size
	apix = float(halfMap1.voxel_size.x);
	if args.apix is not None:
		print('Pixel size set to {:.3f} Angstroem. (Pixel size encoded in map: {:.3f})'.format(args.apix, apix));
		apix = args.apix;
	else:
		print('Pixel size was read as {:.3f} Angstroem. If this is incorrect, please specify with -p pixelSize'.format(apix));
		args.apix = apix;


	# set output filename
	if args.outputFilename is not None:
		splitFilename = os.path.splitext(os.path.basename(args.outputFilename));
	else:
		splitFilename = os.path.splitext(os.path.basename(args.halfmap1));
	outputFilename = splitFilename[0] + "_localResolutions.mrc";


	# handle FDR correction procedure
	if args.method is not None:
		method = args.method;
	else:
		# default is Benjamini-Yekutieli
		method = 'BH';


	#handle window size for local FSC
	if args.window_size is not None:
		wn = args.window_size;
		wn = int(wn);
	else:
		wn = 20;


	#handle step size for local FSC
	if args.stepSize is None:
		stepSize = 2;
	else:
		stepSize = int(args.stepSize);


	#handle number of asymmetric units
	if args.numAsymUnits is None:
		numAsymUnits = 1.0;
	else:
		numAsymUnits = float(args.numAsymUnits);


	#make mask
	if args.mask is not None:
		mask = mrcfile.open(args.mask, mode='r');
		maskData = np.copy(mask.data);
	else:
		print("Using a circular mask ...");
		maskData = FSCutil.makeCircularMask(halfMap1Data, (np.min(halfMap1Data.shape) / 2.0) - 4.0);


	#-------------------------------------------
	#---------- no local Resolutions -----------
	if not args.localResolutions:
		res, FSC, percentCutoffs, threeSigma, threeSigmaCorr, resolution, _ = FSCutil.FSC(halfMap1Data, halfMap2Data,
																					  maskData, apix, 0.143,
																					  numAsymUnits, False, True, None);
		# write the FSC
		FSCutil.writeFSC(res, FSC, percentCutoffs, threeSigmaCorr);
	#-------------------------------------------
	#--------- calc local Resolutions ----------
	else:
		FSCcutoff = 0.5;
		localResMap = FSCutil.localResolutions(halfMap1Data, halfMap2Data, wn, stepSize, FSCcutoff, apix, numAsymUnits,
											   maskData);

		#write the local resolution map
		localResMapMRC = mrcfile.new(outputFilename, overwrite=True);
		localResMap = np.float32(localResMap);
		localResMapMRC.set_data(localResMap);
		localResMapMRC.voxel_size = apix;
		localResMapMRC.close();


	end = time.time();
	totalRuntime = end - start;
	
	print("****** Summary ******");
	print("Runtime: %.2f" %totalRuntime);


if (__name__ == "__main__"):
	main()
