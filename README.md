# Bioimage analysis of cellular changes in human DRG
This repository contains the Python implementation for analyzing DRG tile scan images.

### Outline of analysis steps:
<a href="url"><img src="https://github.com/AmSchulte/DRGhuman/blob/main/Analysis_image.png" width="600" ></a>


1. Preprocessing:
   - with "DataConverter.ipynb"
   - czi images are converted to tiff files, 
   - channels are split and saved seperately 
   - metadata is extracted

2. Annotation:
   - needed for training of a Deep Learning (DL) model
   - with QuPath
   - annotations are exported to json files with "export_json.groovy"
   - json files are converted to masks in shape of the annotated images with "Q Export json_to_mask.ipynb"
   - predicted masks can be viewed with QuPath by converting them to json files ("masks_to_json.ipynb"), which are imported by pull and drop or using "import_json.groovy"

3. Segmentation with Deep Learning (DL):
   - using deepflash2 (https://github.com/matjesg/deepflash2)
   - ground truth estimation and DL model training with "deepflash2_GUI.ipynb"
   - prediction with "Prediction on new data.ipynb"
   - Evaluation of the DL model performance with "Test Comparision Models.ipynb"

4. Thresholding:
   - alternative to DL-based image segmentation for images with good signal to noise ratio
   - the scikit-image filters "unsharp_mask", "threshold_otsu" and "median" are used to create segmentation masks
   - used for FABP7 and IBA1 stainigs of human DRG slices
   
5. Postprocessing:
   - calculate image analysis parameters (e.g. area, intensity, number of neurons) from image+mask datasets using classes in "human DRG.py" for Iba1 ("calculate_Iba1.ipynb") and SGC stainings ("calculate_SGC.ipynb")
   - save the results as json files ("Iba1_results.json", "SGC_results.json")
   - plot the results as boxplots or histogramm ("Iba1 results boxplot.ipynb", "SGC results boxplot.ipynb", "Histogramm Neurons.ipynb")