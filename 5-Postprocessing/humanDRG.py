import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from PIL import Image
from skimage import measure
from skimage.morphology import binary_dilation
import os
from tqdm.auto import tqdm
from scipy import spatial, ndimage
import cv2
from skimage import segmentation


class humanDRG_Iba1:
    def __init__(self, subdir):
        nf_mask_path = os.path.join(subdir, 'NF_pred\masks_adj')
        iba1_mask_path = os.path.join(subdir, 'Iba1_pred\masks_adj')
        
        # default parameters
        self.smallest_roi = 196 
        self.shape = disk(30) #define dilation 
        self.um2_factor = 0.82446
        
        # parameters to calculate
        self.iba1_per_nna_drg = []
        self.neurons_per_npa_drg = []
        self.neuronal_cell_sizes = []
        
        # read matching images of one DRG
        for nf_mask_filename, iba1_mask_filename in tqdm(zip(os.listdir(nf_mask_path), os.listdir(iba1_mask_path))):
            
            # read images
            nf_mask = self.read_img(nf_mask_path, nf_mask_filename)
            iba1_mask = self.read_img(iba1_mask_path, iba1_mask_filename)
            
            # remove rois that are smaller than the smallest annotated size 
            nf_mask_adj = self.remove_small_rois(nf_mask, self.smallest_roi)
            
            #defining the neuron-near area (NNA)
            nna = binary_dilation (nf_mask_adj, self.shape) #dilate nf segmentation for NNA
            neuronarea_um2 = (np.sum (nna)) * self.um2_factor #get neuronarea in um^2
            
            # get values for neurons
            nf_labels = measure.label(nf_mask_adj)
            cell_sizes = np.unique(nf_labels.flatten(), return_counts = True)[1][1:]
            cell_sizes = cell_sizes * self.um2_factor #in um2
            
            neurons_per_npa = self.get_neurons_per_npa(nf_labels, nna, nf_mask_adj)
            
            # get overlap of neuronal polygon area (NPA) and Iba1 mask
            add_masks = nna + iba1_mask
            overlap = (add_masks == 2)*1
            
            # get ratio of iba1 positive area
            iba1_per_nna = np.sum(overlap)/np.sum(nna)
            
            
            self.neuronal_cell_sizes.append(cell_sizes.tolist())
            self.neurons_per_npa_drg.append(neurons_per_npa)            
            self.iba1_per_nna_drg.append(iba1_per_nna)
    
    def read_img(self, directory, filename):
        path = os.path.join(directory, filename)
        return plt.imread(path)     
        
    def remove_small_rois(self, mask, smallest_roi):
        labels = measure.label(mask)
        unique, counts = np.unique(labels, return_counts=True)
        thresholded = unique[counts>smallest_roi][1:]
        filtered = np.isin(labels, thresholded)
        return filtered
    
    def get_neurons_per_npa(self, labels, nna, nf_mask_adj):
        contours = measure.find_contours(nna, 0.9)
        outlines = np.array([j for i in contours for j in i])
        hull = spatial.ConvexHull(outlines)
        
        polygon = []
        for vertice in hull.vertices:
            polygon.append(outlines[vertice])
        polygon = np.array(polygon)
        
        npa_mask = np.zeros(nna.shape[:2])
        npa_mask = cv2.fillPoly(npa_mask, np.array([np.flip(polygon,axis=None)], dtype=np.int32), 1)
        
        return labels.max()/(hull.volume*0.000000824464) #in mm^2


class humanDRG_SGC:
    def __init__(self, subdir):
        nf_mask_path = os.path.join(subdir, 'NF_pred\masks_adj')
        fabp7_mask_path = os.path.join(subdir, 'Fabp7_pred\masks')
        apoj_path = os.path.join(subdir, 'ApoJ')

        # default parameters
        self.smallest_roi = 196 
        self.shape = disk(30) #define dilation 
        self.um2_factor = 0.82446
        
        # parameters to calculate
        self.fabp7_in_nna_drg = []
        self.fabp7_in_npa_drg = []
        self.neurons_per_npa_drg = []
        self.neuronal_cell_sizes = []
        self.ring_ratio_fabp7 = []
        self.near_ratio_fabp7 = []
        self.apoj_intensities_nna = []
        self.apoj_intensities_npa = []

        # read matching images of one DRG
        for nf_mask_filename, fabp7_mask_filename, apoj_img_filename in tqdm(zip(os.listdir(nf_mask_path), os.listdir(fabp7_mask_path), os.listdir(apoj_path))):
            
            # read images
            nf_mask = self.read_img(nf_mask_path, nf_mask_filename)
            fabp7_mask = self.read_img(fabp7_mask_path, fabp7_mask_filename)
            apoj_img = self.read_img(apoj_path, apoj_img_filename)
            
            # remove rois that are smaller than the smallest annotated size 
            nf_mask_adj = self.remove_small_rois(nf_mask, self.smallest_roi)
            
            #defining the neuron-near area (NNA)
            nna = binary_dilation (nf_mask_adj, self.shape) #dilate nf segmentation for NNA
            neuronarea_um2 = (np.sum (nna)) * self.um2_factor #get neuronarea in um^2
            
            # get values for neurons
            nf_labels = measure.label(nf_mask_adj)
            cell_sizes = np.unique(nf_labels.flatten(), return_counts = True)[1][1:]
            cell_sizes = cell_sizes * self.um2_factor #in um2
            number_of_neurons = len(cell_sizes)
            
            npa_mask, neurons_per_npa = self.get_neurons_per_npa(nf_labels, nna, nf_mask_adj)*1


            # get overlap of neuron-near area (NNA) and Fabp7 mask without neurons  
            nna = np.uint8(nna)
            nna_no_neurons = self.get_no_overlap(nna, nf_mask_adj)
            overlap = self.get_overlap(nna_no_neurons, fabp7_mask)
            overlap_fabp7_nna = self.get_overlap(nna, fabp7_mask)

            # get overlap of neuronal polygon area (NPA) and Fabp7 mask   
            npa_mask = np.uint8(npa_mask)
            npa_no_neurons = self.get_no_overlap(npa_mask, nf_mask_adj)
            overlap_fabp7_npa = self.get_overlap(npa_mask, fabp7_mask)
            
            # get ratio of fabp7 positive area
            fabp7_in_nna = np.sum(overlap_fabp7_nna)/np.sum(nna)
            fabp7_in_npa = np.sum(overlap_fabp7_npa)/np.sum(npa_mask)

            # get the ring size (%) of Fabp7 around the neurons 
            fabp7_ring = self.get_rings(nf_labels, fabp7_mask)


            # get the number of rings that are bigger than 0%/50% around the neurons
            number_of_near_fabp7 = len([ring for ring in fabp7_ring if ring>0])
            if number_of_neurons > 0:
                near_fabp7 = (number_of_near_fabp7/number_of_neurons)*100
            
            number_of_rings_fabp7 = len([ring for ring in fabp7_ring if ring>0.5])
            if number_of_neurons > 0:
                ring_fabp7 = (number_of_rings_fabp7/number_of_neurons)*100


            # get ApoJ intensity in Fabp7+ SGCs normalized to intensity in NNA
            no_overlap = self.get_no_overlap(nna_no_neurons, overlap)
            intensity_bg = self.get_intensity_per_area(apoj_img, no_overlap)
            intensity_in_sgc = self.get_intensity_per_area(apoj_img, overlap)
            intensity_apoj_nna = intensity_in_sgc/intensity_bg
         
            # get ApoJ intensity in Fabp7+ SGCs normalized to intensity in NPA
            npa_no_neurons = self.get_no_overlap(npa_mask, nf_mask_adj)
            no_overlap_npa = self.get_no_overlap(npa_no_neurons, overlap)
            intensity_bg = self.get_intensity_per_area(apoj_img, no_overlap_npa)
            intensity_in_sgc = self.get_intensity_per_area(apoj_img, overlap)
            intensity_apoj_npa = intensity_in_sgc/intensity_bg
            
            self.neuronal_cell_sizes.append(cell_sizes.tolist())
            self.neurons_per_npa_drg.append(neurons_per_npa) 
            self.fabp7_in_npa_drg.append(fabp7_in_npa)           
            self.fabp7_in_nna_drg.append(fabp7_in_nna)
            self.ring_ratio_fabp7.append(ring_fabp7)
            self.near_ratio_fabp7.append(near_fabp7)
            self.apoj_intensities_nna.append(intensity_apoj_nna)
            self.apoj_intensities_npa.append(intensity_apoj_npa)
    
    
    def get_neurons_per_npa(self, labels, nna, nf_mask_adj):
        contours = measure.find_contours(nna, 0.9)
        outlines = np.array([j for i in contours for j in i])
        hull = spatial.ConvexHull(outlines)
        
        polygon = []
        for vertice in hull.vertices:
            polygon.append(outlines[vertice])
        polygon = np.array(polygon)
        
        npa_mask = np.zeros(nna.shape[:2])
        npa_mask = cv2.fillPoly(npa_mask, np.array([np.flip(polygon,axis=None)], dtype=np.int32), 1)
        
        # 1px = 0.000000824464mm^2
        return npa_mask, labels.max()/(hull.volume*0.000000824464) #in mm^2
    
    def read_img(self, directory, filename):
        path = os.path.join(directory, filename)
        return plt.imread(path)     
        
    def remove_small_rois(self, mask, smallest_roi):
        labels = measure.label(mask)
        unique, counts = np.unique(labels, return_counts=True)
        thresholded = unique[counts>smallest_roi][1:]
        filtered = np.isin(labels, thresholded)
        return filtered

    def get_overlap(self, mask1, mask2):
        add_masks = mask1 + mask2
        overlap = (add_masks == 2)*1
        return np.uint8(overlap)

    def get_no_overlap(self, mask1, mask2):
        add_masks = mask1 + mask2
        no_overlap = add_masks.copy()
        no_overlap[add_masks == 2] = 0    
        return np.uint8(no_overlap)

    def get_intensity_per_area(self, image, mask):
        masking = mask>0
        masked = image*masking
        return np.sum(np.uint64(masked))/np.sum(mask)

    def get_rings(self, nf_label, mask): 

        # add 5 black pixels to border of ring mask to make calculation possible for neurons at border of image
        mask_dil = np.zeros([mask.shape[0]+10, mask.shape[1]+10])
        mask_dil[5:-5, 5:-5] = mask

        #dilate mask by one pixel
        mask_dilated = ndimage.binary_dilation(mask_dil)*1

        # add 5 black pixels to border of image to make calculation possible for neurons at border of image
        nf_label_dilated = np.zeros([nf_label.shape[0]+10, nf_label.shape[1]+10])
        nf_label_dilated[5:-5, 5:-5] = nf_label
        nf_label_dilated = np.int64(nf_label_dilated)

        rings = []

        for i in range(nf_label_dilated.max()+1):
            if i > 0:
                #get single neuron        
                nf_object = nf_label_dilated==i
                cell_pre = nf_object*1
                #dilate neuron by one pixel
                cell = ndimage.binary_dilation(cell_pre)

                # cut neuron and ring image by 5 pixels around the neuron
                cellbounds = np.where(nf_label_dilated==i)
                x_min = cellbounds[0].min()-5
                x_max = cellbounds[0].max()+5
                y_min = cellbounds[1].min()-5
                y_max = cellbounds[1].max()+5
                single_cell = cell[x_min:x_max,y_min:y_max]

                ring = mask_dilated[x_min:x_max,y_min:y_max]

                # find edges of neuron
                edges = segmentation.find_boundaries(single_cell)
                nf_border = edges*single_cell

                # count pixels of edges of neuron
                all_edges = np.ndarray.flatten(nf_border)
                colors, counts = np.unique(all_edges, return_counts = True, axis = 0)
                nf_border_count = counts[1]

                # get overlap of neuron and ring and count the pixels
                overlap = ring+nf_border==2
                all_overlap = np.ndarray.flatten(overlap)
                colors, counts = np.unique(all_overlap, return_counts = True, axis = 0)

                # calculate ring size (overlap/edges)
                if len(counts) ==1:
                    ring_size = 0
                else: 
                    overlap_count = counts[1]
                    ring_size = overlap_count/nf_border_count
                rings.append(ring_size) 
        return rings

