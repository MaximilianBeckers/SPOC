# False discovery rate control of cryo-EM maps

Confidence maps are complementary maps generated from cryo-EM maps by means of statistical hypothesis testing and subsequent FDR control. They allow thresholding of EM maps based on the expected amount of background noise visible at the respective threshold and thus allow rigorous error assessment of visible features in the density. 
Additional post-processing like local filtering or local amplitude scaling (LocScale) can be incorporated in the framework in order to increase the statistical power.

**For a detailed introduction, please see the file confidenceMaps_tutorial.pdf**

## Getting Started


### Prerequisites

The software is written in Python and itself dependent on NumPy and Matplotlib libraries. 


### Installing

The folder just has to be copied to your computer and can be run with a NumPy and Matplotlib capable Python installation.


## How to use

The simplest, and probably most important case, is the generation of a confidence map from a simple cryoEM density without incorporation of local resolution or atomic model information.

```
python /path/to/FDRcontrol.py -em yourMap.mrc -p thePixelSize
```

The output will be the corresponding confidence map ( yourMap_confidenceMap.mrc ) and a diagnostic image (diag_image.pdf), that shows three slices thorugh the map together with the regions used for noise estimation. In order to get good estimates of the background noise distribution, you should make sure that the region contains just noise and no signal of the particle.

Size of the noise estimation region can be adjusted with -w sizeOfRegion. If the default regions for noise estimation fall into noise, you can specify the center of region of your choice with -noiseBox x y z .
For example:

```
python /path/to/FDRcontrol.py -em yourMap.mrc -p 5.6 -w 20 -noiseBox 50 50 120
```

### Incorporation of local resolution information

A corresponding local resolution map can be supplied to the program by -locResMap yourLocalResolutionMap.mrc .

The ouput will be a locally filtered map together with the diagnostic image and the confidence map. Adjustment of the noise estimation region can be done accordingly.

Example usage:
```
python /path/to/FDRcontrol.py -em yourMap.mrc -p thePixelSize -locResMap yourLocalResolutionMap.mrc
```

### Incorporation of local amplitude scaling

LocScale inside FDRcontrol.py can be used by supplying a model map with -mm youModelMap.mrc, where the model map is generated in the LocScale workflow. For usage of parallelisation, -mpi can be specified. The window size can be specified with -w windowSize,
 and is also the size of the region used for noise estimation.
The ouput will be the locally scaled map together with the diagnostic image and the confidence map. 

Example usage:

```
python /path/to/FDRcontrol.py -em yourMap.mrc -p thePixelSize -mm yourModelMap.mrc -w 20
```

## Instructions for use and important tips

The programs require an unmasked map as input. Masking will make the noise estimation impossible.

It is critical that the regions used for noise estimation do not fall into the particle of interest. While in single particle analysis (SPA), 
the data generating process makes choice of the regions straightforward, sub-tomogram-averaged structures usually require specification of this region from the user.
The region can be identified by means of the slice-views in the diagnostic image of a first run with default parameters, and then adjust center and width of the noise region for a subsequent run. 
The axes in the diagnostic image are labelled with the voxel indices.

**Always have a look in the diagnostic image.** 

If your map contains large background areas, i.e. small particle compared to the box size, you can try to increase the sizes of the noise estimation regions with -w in 
order to get more accurate estimates of the background noise distribution.

Be aware, inclusion of either local resolution and/or atomic model information depend on the accuracy of this prior information, if the information is inaccurate, the final confidence maps will be, too.
Make sure orientations of model maps and/or local resolution maps with respect to the input EM-map are correct.

**As confidence maps are almost binary and have huge contrast, they might look quite sharp and edged. For visualization the representation can be easily improved, for example in Chimera by 
turning on surface smoothing in the Volume Viewer in Features --> Sufrace and Mesh options.**

## Demonstration with TRPV1 EMD5778

We will demonstrate the procedure using the 3.4 Angstrom structure of TRPV1 (EMD5778, Liao et al. 2013)

Download the map by clicking on: 

ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-5778/other/TRPV1_sharpened_-100_3.4A.map.gz

Go into the respective directory and unzip the file. 

```
gunzip TRPV1_sharpened_-100_3.4A.map.gz
```

We are ready to generate the confidence map by

```
python /path/to/FDRcontrol.py -em TRPV1_sharpened_-100_3.4A.map -p 1.2156
```

The expected run time on normal desktop computer should be around 1-2 minutes for this example.

    
    



