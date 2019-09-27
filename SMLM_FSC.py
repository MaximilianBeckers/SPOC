# Author: Maximilian Beckers, EMBL Heidelberg, Sachse Group (2019)

# import some stuff
from SMLMUtil import SMLM
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import argparse, sys
import time

# *************************************************************
# ****************** Commandline input ************************
# *************************************************************

cmdl_parser = argparse.ArgumentParser(
	prog=sys.argv[0],
	description='*** Thresholding of FSC curves by FDR control - Single Molecule localization ***',
	formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30), add_help=True);

cmdl_parser.add_argument('-localizations', '--localizations', metavar="localizations.csv", type=str, required=False,
						 help='Input filename localizations');
cmdl_parser.add_argument('-image1', '--image1', metavar="image_1.jpg", type=str, required=False,
						 help='Input filename of image 1');
cmdl_parser.add_argument('-image2', '--image2', metavar="image_2.jpg", type=str, required=False,
						 help='Input filename of image 2');
cmdl_parser.add_argument('-size', '--size', type=int, required=False, default=5000,
						 help='size of image for binning the localizations (default: 5000)');
cmdl_parser.add_argument('-localResolutions', '--localResolutions', type=bool, required=False,
						 help='calculation of local resolutions', default=False);
cmdl_parser.add_argument('-w', '--window_size', metavar="windowSize", type=int, required=False,
						 help="Input window size for local resolution estimation (default: 50)", default=50);
cmdl_parser.add_argument('-stepSize', '--stepSize', type=int, required=False,
						 help="Pixels to skip for local resolution estimation (default: 5)", default=5);

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

	if args.size is not None:
		imageSize = args.size;


	if args.localResolutions:
		stepSize = args.stepSize;
		boxSize = args.window_size;

	if args.localizations is not None:

		#read the localizations
		localizations = np.loadtxt(args.localizations, delimiter= ",", skiprows=1, usecols=(0, 1));

		SMLMObject = SMLM.SMLM();
		SMLMObject.resolution(localizations, None, None, imageSize);

	else:

		#read the two images
		if (args.image1 is None) or (args.image2 is None):
			print("One of the two images for correlation is missing. Exit ...");
			sys.exit();
		else:
			image1 = ndimage.imread(args.image1);
			image2 = ndimage.imread(args.image2);

		SMLMObject = SMLM.SMLM();
		SMLMObject.resolution(None, image1, image2, imageSize);

	#************************
	#***** plot images ******
	#************************

	plt.imshow(SMLMObject.fullMap.T, cmap='hot', origin='lower')
	plt.colorbar();
	plt.savefig('heatMap_full_png', dpi=300);
	plt.close();

	plt.imshow(SMLMObject.filteredMap.T, cmap='hot', origin='lower')
	plt.colorbar();
	plt.savefig('heatMap_filt.png', dpi=300);
	plt.close();


if (__name__ == "__main__"):
	main()
