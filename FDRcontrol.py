# Author: Maximilian Beckers, EMBL Heidelberg, Sachse Group (2017)

# import some stuff
from confidenceMapUtil import FDRutil, confidenceMapMain
import numpy as np
import argparse, os, sys
import subprocess
import math
import gc
import os.path
import time
import sys
import mrcfile

# *************************************************************
# ****************** Commandline input ************************
# *************************************************************

cmdl_parser = argparse.ArgumentParser(
	prog=sys.argv[0],
	description='*** Analyse density ***',
	formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-em', '--em_map', metavar="em_map.mrc", type=str, required=True,
						 help='Input filename EM map');
cmdl_parser.add_argument('-halfmap2', '--halfmap2', metavar="halfmap2.mrc", type=str, required=False,
						 help='Input filename halfmap 2');
cmdl_parser.add_argument('-p', '--apix', metavar="apix", type=float, required=False,
						 help='pixel Size of input map');
cmdl_parser.add_argument('-fdr', '--fdr', metavar="fdr", type=float, required=False,
						 help='False Discovery Rate');
cmdl_parser.add_argument('-locResMap', metavar="locResMap.mrc", type=str, required=False,
						 help='Input local Resolution Map');
cmdl_parser.add_argument('-method', metavar="method", type=str, required=False,
						 help="Method for multiple testing correction. 'BY' for Benjamini-Yekutieli, 'BH' for Benjamini-Hochberg or 'Holm' for Holm FWER control");
cmdl_parser.add_argument('-mm', '--model_map', metavar="model_map.mrc", type=str, required=False,
						 help="Input model map for model based amplitude scaling");
cmdl_parser.add_argument('-w', '--window_size', metavar="windowSize", type=float, required=False,
						 help="Input window size for local Amplitude scaling and background noise estimation");
cmdl_parser.add_argument('-mpi', action='store_true', default=False,
						 help="Set this flag if MPI should be used for the local amplitude scaling");
cmdl_parser.add_argument('-o', '--outputFilename', metavar="output.mrc", type=str, required=False,
						 help="Name of the output");
cmdl_parser.add_argument('-noiseBox', metavar="[x, y, z]", nargs='+', type=int, required=False,
						 help="Box coordinates for noise estimation");
cmdl_parser.add_argument('-meanMap', '--meanMap', type=str, required=False,
						 help="3D map of noise means to be used for FDR control");
cmdl_parser.add_argument('-varianceMap', '--varianceMap', type=str, required=False,
						 help="3D map of noise variances to be used for FDR control");
cmdl_parser.add_argument('-testProc', '--testProc', type=str, required=False,
						 help="choose between right, left and two-sided testing");
cmdl_parser.add_argument('-lowPassFilter', '--lowPassFilter', type=float, required=False,
						 help="Low-pass filter the map at the given resoultion prior to FDR control");
cmdl_parser.add_argument('-ecdf', action='store_true', default=False,
						 help="Set this flag if the empricical cumulative distribution function should be used instead of the standard normal distribution");
cmdl_parser.add_argument('-w_locscale', '--window_size_locscale', metavar="windowSize_locScale", type=float,
						 required=False,
						 help="Input window size for local Amplitude scaling");
cmdl_parser.add_argument('-stepSize', '--stepSize', metavar="stepSize_locScale", type=int, required=False,
						 help="Voxels to skip for local amplitude scaling");


# ************************************************************
# ********************** main function ***********************
# ************************************************************


def main():
	start = time.time();

	# get command line input
	args = cmdl_parser.parse_args();

	# no ampltidue scaling will be done
	print('************************************************');
	print('******* Significance analysis of EM-Maps *******');
	print('************************************************');

	# if varianceMap is given, use it
	if args.varianceMap is not None:
		varMap = mrcfile.open(args.varianceMap, mode='r');
		varMapData = np.copy(varMap.data);
	else:
		varMapData = None;

	# if meanMap is given, use it
	if args.meanMap is not None:
		meanMap = mrcfile.open(args.meanMap, mode='r');
		meanMapData = np.copy(meanMap.data);
	else:
		meanMapData = None;

	# load the maps
	if args.halfmap2 is not None:
		if args.em_map is None:
			print("One half map missing! Exit ...")
			sys.exit();
		else:
			# load the maps
			filename = args.em_map;
			map1 = mrcfile.open(args.em_map, mode='r');
			apix = float(map1.voxel_size.x);
			halfMapData1 = np.copy(map1.data);
			sizeMap = halfMapData1.shape;

			map2 = mrcfile.open(args.halfmap2, mode='r');
			halfMapData2 = np.copy(map2.data);

			print("Estimating local noise levels ...")
			varMapData = FDRutil.estimateNoiseFromHalfMaps(halfMapData1, halfMapData2, 20, 2);
			meanMapData = np.zeros(varMapData.shape)

			mapData = (halfMapData1 + halfMapData2) * 0.5;
			halfMapData1 = 0;
			halfMapData2 = 0;

	else:
		# load single map
		filename = args.em_map;
		map = mrcfile.open(filename, mode='r');
		apix = float(map.voxel_size.x);
		mapData = np.copy(map.data);

	if args.apix is not None:
		print('Pixel size set to {:.3f} Angstroem. (Pixel size encoded in map: {:.3f})'.format(args.apix, apix));
		apix = args.apix;
	else:
		print('Pixel size was read as {:.3f} Angstroem. If this is incorrect, please specify with -p pixelSize'.format(
			apix));
		args.apix = apix;

	# set output filename
	if args.outputFilename is not None:
		splitFilename = os.path.splitext(os.path.basename(args.outputFilename));
	else:
		splitFilename = os.path.splitext(os.path.basename(filename));


	# if local resolutions are given, use them
	if args.locResMap is not None:
		locResMap = mrcfile.open(args.locResMap, mode='r');
		locResMapData = np.copy(locResMap.data);
	else:
		locResMapData = None;

	# get LocScale input
	if args.model_map is not None:
		modelMap = mrcfile.open(args.model_map, mode='r');
		modelMapData = np.copy(modelMap.data);
	else:
		modelMapData = None;

	if args.stepSize is not None:
		stepSize = args.stepSize;
	else:
		stepSize = None;

	if args.window_size_locscale is not None:
		windowSizeLocScale = args.window_size_locscale;
	else:
		windowSizeLocScale = None;

	if args.mpi:
		mpi = True;
	else:
		mpi = False;

	if (args.stepSize is not None) & (args.window_size_locscale is not None):
		if args.stepSize > args.window_size_locscale:
			print("Step Size cannot be bigger than the window_size. Job is killed ...")
			return;

	# run the actual analysis
	confidenceMap, locFiltMap, locScaleMap, mean, var = confidenceMapMain.calculateConfidenceMap(mapData, apix,
																								 args.noiseBox,
																								 args.testProc,
																								 args.ecdf,
																								 args.lowPassFilter,
																								 args.method,
																								 args.window_size,
																								 locResMapData,
																								 meanMapData,
																								 varMapData, args.fdr,
																								 modelMapData, stepSize,
																								 windowSizeLocScale,
																								 mpi);

	if locFiltMap is not None:
		locFiltMapMRC = mrcfile.new(splitFilename[0] + '_locFilt.mrc', overwrite=True);
		locFiltMap = np.float32(locFiltMap);
		locFiltMapMRC.set_data(locFiltMap);
		locFiltMapMRC.voxel_size = apix;
		locFiltMapMRC.close();

	if locScaleMap is not None:
		locScaleMapMRC = mrcfile.new(splitFilename[0] + '_scaled.mrc', overwrite=True);
		locScaleMap = np.float32(locScaleMap);
		locScaleMapMRC.set_data(locScaleMap);
		locScaleMapMRC.voxel_size = apix;
		locScaleMapMRC.close();

	"""if (locScaleMap is not None) | (locFiltMap is not None):
		meanMapMRC = mrcfile.new(splitFilename[0] + '_mean.mrc', overwrite=True);
		mean = np.float32(mean);
		meanMapMRC.set_data(mean);
		meanMapMRC.voxel_size = apix;
		meanMapMRC.close();
		varMapMRC = mrcfile.new(splitFilename[0] + '_var.mrc', overwrite=True);
		var = np.float32(var);
		varMapMRC.set_data(var);
		varMapMRC.voxel_size = apix;
		varMapMRC.close();"""

	# write the confidence Maps
	confidenceMapMRC = mrcfile.new(splitFilename[0] + '_confidenceMap.mrc', overwrite=True);
	confidenceMap = np.float32(confidenceMap);
	confidenceMapMRC.set_data(confidenceMap);
	confidenceMapMRC.voxel_size = apix;
	confidenceMapMRC.close();

	# write the confidence Maps
	confidenceMapMRC = mrcfile.new(splitFilename[0] + '_confidenceMap_-log10FDR.mrc', overwrite=True);
	confidenceMap = np.float32(1.0 - confidenceMap);
	confidenceMap[confidenceMap == 0] = 0.0000000001;
	confidenceMapMRC.set_data(-np.log10(confidenceMap));
	confidenceMapMRC.voxel_size = apix;
	confidenceMapMRC.close();

	end = time.time();
	totalRuntime = end - start;

	FDRutil.printSummary(args, totalRuntime);


if (__name__ == "__main__"):
	main()

	







