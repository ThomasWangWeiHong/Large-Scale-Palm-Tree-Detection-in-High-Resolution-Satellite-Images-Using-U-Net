import cv2
import glob
import json
import numpy as np
import rasterio
from keras import backend as K
from keras.models import Input, Model
from keras.layers import BatchNormalization, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam



def training_mask_generation(input_image_filename, input_geojson_filename):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        image = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
        
    mask = np.zeros((image.shape[0], image.shape[1]))
    
    ulx = metadata['transform'][2]
    xres = metadata['transform'][0]
    uly = metadata['transform'][5]
    yres = metadata['transform'][4]
                                      
    lrx = ulx + (image.shape[1] * xres)                                                         
    lry = uly - (image.shape[0] * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((image.shape[1]) ** 2 / (image.shape[1] + 1)) / (lrx - ulx)
        yf = ((image.shape[0]) ** 2 / (image.shape[0] + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask, position, 1)
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, image_height_size, image_width_size, mode, percentage_overlap, 
                                      buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image and its mask in the form of fixed size segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((image_array.shape[0], x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = mask_array
        
    img_list = []
    mask_list = []
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, 0]
            mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
            mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
            mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
            mask_flip_hor = cv2.flip(mask_original, 0)
            mask_flip_vert = cv2.flip(mask_original, 1)
            mask_flip_both = cv2.flip(mask_original, -1)
            mask_list.extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, mask_flip_vert, 
                              mask_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(mask_list), image_height_size, image_width_size, 1))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        mask_segment_array[index, :, :, 0] = mask_list[index]
        
    return image_segment_array, mask_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\' + 'Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Training Polygons' + '\\Train_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    
    for file in range(len(img_files)):
        with rasterio.open(img_files[file]) as f:
            metadata = f.profile
            img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
            
        mask = training_mask_generation(img_files[file], polygon_files[file])
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 0, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 1, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 2, 
                                                                      percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 3, 
                                                                      percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    
    return img_full_array, mask_full_array



def dice_coef(y_true, y_pred):
    """ 
    This function generates the dice coefficient for use in semantic segmentation model training. 
    
    """
    
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    coef = (2 * intersection) / (K.sum(y_true_flat) + K.sum(y_pred_flat))
    
    return coef



def dice_coef_loss(y_true, y_pred):
    """ 
    This function generates the dice coefficient loss function for use in semantic segmentation model training. 
    
    """
    
    return -dice_coef(y_true, y_pred)



def unet_a(img_height_size, img_width_size, n_bands, layer_1_filters = 32, l_r = 0.0001):
    """
    This function is used to generate the U - Net - A architecture as described in the paper 'Large Scale Palm Tree Detection 
    in High Resolution Satellite Images Using U - Net' by Freudenberg M., Nolke N., Agostini A., Urban K., Worgotter F., 
    Kleinn C. (2019)
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - layer_1_filters: Number of feature maps to be generated in the first layer of the U - Net - A model
    - l_r: Learning rate to be applied for the Adam optimizer
    
    Outputs:
    - unet_a_model: U - Net - A model to be trained using input parameters and network architecture
    
    """
    
    layer_2_filters = int(2 * layer_1_filters)
    layer_3_filters = int(2 * layer_2_filters)
    layer_4_filters = int(2 * layer_3_filters)
    layer_5_filters = int(2 * layer_4_filters)
    
    
    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    
    
    conv_1_1 = Conv2D(layer_1_filters, (3, 3), padding = 'same', activation = 'relu')(img_input)
    conv_1_1_bn = BatchNormalization()(conv_1_1)
    conv_1_2 = Conv2D(layer_1_filters, (3, 3), padding = 'same', activation = 'relu')(conv_1_1_bn)
    conv_1_2_bn = BatchNormalization()(conv_1_2)
    
    
    conv_2_1_input = MaxPooling2D()(conv_1_2_bn)
    conv_2_1 = Conv2D(layer_2_filters, (3, 3), padding = 'same', activation = 'relu')(conv_2_1_input)
    conv_2_1_bn = BatchNormalization()(conv_2_1)
    conv_2_2 = Conv2D(layer_2_filters, (3, 3), padding = 'same', activation = 'relu')(conv_2_1_bn)
    conv_2_2_bn = BatchNormalization()(conv_2_2)
    
    
    conv_3_1_input = MaxPooling2D()(conv_2_2_bn)
    conv_3_1 = Conv2D(layer_3_filters, (3, 3), padding = 'same', activation = 'relu')(conv_3_1_input)
    conv_3_1_bn = BatchNormalization()(conv_3_1)
    conv_3_2 = Conv2D(layer_3_filters, (3, 3), padding = 'same', activation = 'relu')(conv_3_1_bn)
    conv_3_2_bn = BatchNormalization()(conv_3_2)
    
    
    conv_4_1_input = MaxPooling2D()(conv_3_2_bn)
    conv_4_1 = Conv2D(layer_4_filters, (3, 3), padding = 'same', activation = 'relu')(conv_4_1_input)
    conv_4_1_bn = BatchNormalization()(conv_4_1)
    conv_4_2 = Conv2D(layer_4_filters, (3, 3), padding = 'same', activation = 'relu')(conv_4_1_bn)
    conv_4_2_bn = BatchNormalization()(conv_4_2)
    
    
    conv_5_1_input = MaxPooling2D()(conv_4_2_bn)
    conv_5_1 = Conv2D(layer_5_filters, (3, 3), padding = 'same', activation = 'relu')(conv_5_1_input)
    conv_5_1_bn = BatchNormalization()(conv_5_1)
    conv_5_2 = Conv2D(layer_5_filters, (3, 3), padding = 'same', activation = 'relu')(conv_5_1_bn)
    conv_5_2_bn = BatchNormalization()(conv_5_2)
    
    
    upsam_5 = UpSampling2D()(conv_5_2_bn)
    upsam_4_1_input = concatenate([upsam_5, conv_4_2_bn])
    upsam_4_1 = Conv2D(layer_4_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_4_1_input)
    upsam_4_1_bn = BatchNormalization()(upsam_4_1)
    upsam_4_2 = Conv2D(layer_4_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_4_1_bn)
    upsam_4_2_bn = BatchNormalization()(upsam_4_2)
    
    
    upsam_4 = UpSampling2D()(upsam_4_2_bn)
    upsam_3_1_input = concatenate([upsam_4, conv_3_2_bn])
    upsam_3_1 = Conv2D(layer_3_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_3_1_input)
    upsam_3_1_bn = BatchNormalization()(upsam_3_1)
    upsam_3_2 = Conv2D(layer_3_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_3_1_bn)
    upsam_3_2_bn = BatchNormalization()(upsam_3_2)
    
    
    upsam_3 = UpSampling2D()(upsam_3_2_bn)
    upsam_2_1_input = concatenate([upsam_3, conv_2_2_bn])
    upsam_2_1 = Conv2D(layer_2_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_2_1_input)
    upsam_2_1_bn = BatchNormalization()(upsam_2_1)
    upsam_2_2 = Conv2D(layer_2_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_2_1_bn)
    upsam_2_2_bn = BatchNormalization()(upsam_2_2)
    
    
    upsam_2 = UpSampling2D()(upsam_2_2_bn)
    upsam_1_1_input = concatenate([upsam_2, conv_1_2_bn])
    upsam_1_1 = Conv2D(layer_1_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_1_1_input)
    upsam_1_1_bn = BatchNormalization()(upsam_1_1)
    upsam_1_2 = Conv2D(layer_1_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_1_1_bn)
    upsam_1_2_bn = BatchNormalization()(upsam_1_2)
    
    
    pred_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(upsam_1_2_bn)
    
    
    unet_a_model = Model(inputs = img_input, outputs = pred_layer)
    unet_a_model.compile(loss = dice_coef_loss, optimizer = Adam(lr = l_r), metrics = [dice_coef])
    
    return unet_a_model



def unet_b(img_height_size, img_width_size, n_bands, layer_1_filters = 32, l_r = 0.0001):
    """
    This function is used to generate the U - Net - B architecture as described in the paper 'Large Scale Palm Tree Detection 
    in High Resolution Satellite Images Using U - Net' by Freudenberg M., Nolke N., Agostini A., Urban K., Worgotter F., 
    Kleinn C. (2019)
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - layer_1_filters: Number of feature maps to be generated in the first layer of the U - Net - B model
    - l_r: Learning rate to be applied for the Adam optimizer
    
    Outputs:
    - unet_b_model: U - Net - B model to be trained using input parameters and network architecture
    
    """
    
    layer_2_filters = int(2 * layer_1_filters)
    layer_3_filters = int(2 * layer_2_filters)
    layer_4_filters = int(2 * layer_3_filters)
    
    
    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    
    
    conv_1 = Conv2D(layer_1_filters, (3, 3), padding = 'same', activation = 'relu')(img_input)
    conv_1_bn = BatchNormalization()(conv_1)
    
    
    conv_2_input = MaxPooling2D()(conv_1_bn)
    conv_2 = Conv2D(layer_2_filters, (3, 3), padding = 'same', activation = 'relu')(conv_2_input)
    conv_2_bn = BatchNormalization()(conv_2)
    
    
    conv_3_input = MaxPooling2D()(conv_2_bn)
    conv_3 = Conv2D(layer_3_filters, (3, 3), padding = 'same', activation = 'relu')(conv_3_input)
    conv_3_bn = BatchNormalization()(conv_3)
    
    
    conv_4_input = MaxPooling2D()(conv_3_bn)
    conv_4 = Conv2D(layer_4_filters, (3, 3), padding = 'same', activation = 'relu')(conv_4_input)
    conv_4_bn = BatchNormalization()(conv_4)
    
    
    upsam_4 = UpSampling2D()(conv_4_bn)
    upsam_3_input = concatenate([upsam_4, conv_3_bn])
    upsam_3 = Conv2D(layer_3_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_3_input)
    upsam_3_bn = BatchNormalization()(upsam_3)
    
    
    upsam_3 = UpSampling2D()(upsam_3_bn)
    upsam_2_input = concatenate([upsam_3, conv_2_bn])
    upsam_2 = Conv2D(layer_2_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_2_input)
    upsam_2_bn = BatchNormalization()(upsam_2)
    
    
    upsam_2 = UpSampling2D()(upsam_2_bn)
    upsam_1_input = concatenate([upsam_2, conv_1_bn])
    upsam_1 = Conv2D(layer_1_filters, (3, 3), padding = 'same', activation = 'relu')(upsam_1_input)
    upsam_1_bn = BatchNormalization()(upsam_1)
    
    
    pred_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(upsam_1_bn)
    
    
    unet_b_model = Model(inputs = img_input, outputs = pred_layer)
    unet_b_model.compile(loss = dice_coef_loss, optimizer = Adam(lr = l_r), metrics = [dice_coef])
    
    return unet_b_model



def image_model_predict(input_image_filename, output_filename, img_height_size, img_width_size, fitted_model, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for prediction. The 
    output mask is then allocated to its corresponding location in the image in order to obtain the complete mask for the 
    entire image without being constrained by image size. 
    
    Inputs:
    - input_image_filename: File path of image file for which prediction is to be conducted
    - output_filename: File path of output predicted binary raster mask file
    - img_height_size: Height of image patches to be used for model prediction
    - img_height_size: Width of image patches to be used for model prediction
    - fitted_model: Trained keras model which is to be used for prediction
    - write: Boolean indicating whether to write predicted binary raster mask to file
    
    Output:
    - mask_complete: Numpy array of predicted binary raster mask for input image
    
    """
    
    with rasterio.open(input_image_filename) as f:
        metadata = f.profile
        img = np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0])
     
    y_size = ((img.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((img.shape[1] // img_width_size) + 1) * img_width_size
    
    if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, img.shape[1], img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((img.shape[0], x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    else:
         img_complete = img
            
    mask = np.zeros((img_complete.shape[0], img_complete.shape[1], 1))
    img_holder = np.zeros((1, img_height_size, img_width_size, img.shape[2]))
    
    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img.shape[2]]
            preds = fitted_model.predict(img_holder)
            mask[i : i + img_height_size, j : j + img_width_size, 0] = preds[0, :, :, 0]
            
    mask_complete = np.expand_dims(mask[0 : img.shape[0], 0 : img.shape[1], 0], axis = 2)
    mask_complete = np.transpose(mask_complete, [2, 0, 1]).astype('float32')
    
    
    if write:
        metadata['count'] = 1
        metadata['dtype'] = 'float32'
        
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(mask_complete)
    
    return mask_complete