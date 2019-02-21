from EMAN2 import EMData, EMNumPy


#--------------------------------------------------------------------------------------------------
def readAndFlattenImageStack(filename):
	
	#***********************************************
	#* read imageStack and return Nx(nX*nY) array **
	#***********************************************

	imageStack = EMData();
	imageStack.read_image(filename);
	nx, ny, numImages = imageStack.get_xsize(), imageStack.get_ysize(), imageStack.get_zsize();
	imageStackData = EMNumPy.em2numpy(imageStack);
	imageStack = []; #free memory

	#flatten each image and append to array of NxD, with N the number of images and D the size of the flat image
	imageStackData = np.reshape(imageStackData, (nx*ny, numImages), order='F' );		
	imageStackData = np.transpose(imageStackData);
	print(imageStackData.shape);

	return imageStackData;


#---------------------------------------------------------------------------------
def addNoiseToMapSolvent(map, varNoise):

	#*********************************
	#****** add Noise To Map *********
	#*********************************

	mapData = np.copy(map);
	mapSize = mapData.shape;
	noiseMap = np.random.randn(mapSize[0], mapSize[1], mapSize[2])*math.sqrt(varNoise);	

	mask = EMData();	
	mask.set_size(mapSize[0], mapSize[1], mapSize[2]);
	mask.to_zero(); 
	sphere_radius = (np.min(mapSize)/2.0 - 60); 
	mask.process_inplace("testimage.circlesphere", {"radius":sphere_radius});
	maskData = np.copy(EMNumPy.em2numpy(mask));
	maskData[maskData > 0] = 10;
	maskData[maskData <= 0] = 0;
	maskData[maskData == 0] = 1;
	maskData[maskData == 10] = 0;
	noiseMap = noiseMap*maskData;

	mapData = mapData + noiseMap;

	return mapData; 

#---------------------------------------------------------------------------------------------------
def padFourier(map, apix, finalPix):
	
	sizeMap = np.array([map.get_xsize(), map.get_ysize(), map.get_zsize()]);
	
	#size of the map
	lengthMap = sizeMap[1] * apix;
	finalSize = int(lengthMap/float(finalPix));
	diffSize = finalSize - sizeMap[1];

	if diffSize % 2 != 0 :
		finalSize = finalSize + 1;
		diffSize = finalSize - sizeMap[1];
	
	newPixSize = lengthMap/float(finalSize);
	print('The new pixelSize is: {}'.format(newPixSize));
	
	halfDiffSize = int(diffSize/2.0);

	#do fourier padding
	mapData = np.copy(EMNumPy.em2numpy(map));
	fftMap = np.fft.fftn(mapData);
	fftMapshift = np.fft.fftshift(fftMap);
	
	fftMapPad = np.zeros((finalSize, finalSize, finalSize), dtype=np.complex_);
	fftMapPad[halfDiffSize:(halfDiffSize + sizeMap[1]), halfDiffSize:(halfDiffSize + sizeMap[1]), halfDiffSize:(halfDiffSize + sizeMap[1])] = np.copy(fftMapshift);
	mapPadData = np.fft.ifftn(np.fft.ifftshift(fftMapPad)).real;
	
	mapPad = EMNumPy.numpy2em(mapPadData);

	return mapPad;
