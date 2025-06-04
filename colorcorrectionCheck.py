from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt

root = ctk.CTk()
folder_path = ctk.filedialog.askopenfilename()

def ExtractLeafTips(inDirectoryPath) -> str:

        # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    extensions = ('.jpg', '.png')

    img_path = inDirectoryPath

    #outDirectoryPath = os.path.join(inDirectoryPath, 'processed_images').replace("/", "\\") # Elimintes confusion with python escape codes
    #os.makedirs(outDirectoryPath, exist_ok=True)
    #print(f'In Path: {inDirectoryPath}\nOut Path: {outDirectoryPath}')
    #image_files = pcv.io.read_dataset(inDirectoryPath)

    ext = os.path.splitext(img_path)[-1].lower()
    if ext in extensions:
        # Process each image as before
        args = WorkflowInputs(
            images=[img_path],
            names=img_path,
            result='',
            writeimg=True,
            outdir=rf'{os.getcwd()}',
            debug='',
            sample_label=''
        )
        pcv.params.line_thickness = 2

        # Debug Params
        pcv.params.debug = args.debug
        img, filename, path = pcv.readimage(filename=img_path)

        hh, ww = img.shape[:2]
        maxdim = max(hh, ww)

        # illumination normalize
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # separate channels
        y, cr, cb = cv2.split(ycrcb)

        # get background which paper says (gaussian blur using standard deviation 5 pixel for 300x300 size image)
        # account for size of input vs 300
        sigma = int(5 * maxdim / 300)
        #print('sigma: ',sigma)
        gaussian = cv2.GaussianBlur(y, (3, 3), sigma, sigma)

        # subtract background from Y channel
        y = (y - gaussian + 100)

        # merge channels back
        ycrcb = cv2.merge([y, cr, cb])

        #convert to BGR
        output = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        corrected_img1 = pcv.white_balance(img=output, mode='hist')
        corrected_img2 = pcv.white_balance(img=corrected_img1, mode='hist')

        #gamma corrected
        img_g = pcv.transform.gamma_correct(img=img, gamma=1, gain=1)
        output_g = pcv.transform.gamma_correct(img=output, gamma=2.5, gain=5)
        #output_g=pcv.white_balance(output_g, mode='hist')
        #output_g = pcv.white_balance(img=output_g, mode='hist')
        corrected1_g = pcv.transform.gamma_correct(img=corrected_img1, gamma=1, gain=1)
        corrected2_g = pcv.transform.gamma_correct(img=corrected_img2, gamma=1, gain=1)


        lab = cv2.cvtColor(output_g, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Method 2: HSV thresholding
        hsv = cv2.cvtColor(enhanced_clahe, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_green, upper_green)
        cv2.imshow(None, mask_hsv)
        cv2.waitKey(0)
        

        a_gray2_c = pcv.rgb2gray_cmyk(rgb_img=output, channel='y')
        bin_mask1_2_c = pcv.threshold.otsu(gray_img=a_gray2_c, object_type='dark')
        bin_mask1_2_invert = pcv.invert(bin_mask1_2_c)
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2_invert, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_21 = pcv.opening(clean_mask1_2)

        a_gray2_l = pcv.rgb2gray_lab(rgb_img=output, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2_l, object_type='dark')
        #cv2.imshow("Threshold", bin_mask1_2)
        #cv2.waitKey(0)
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_21 = pcv.opening(clean_mask1_2)

        a_gray2_c_g = pcv.rgb2gray_cmyk(rgb_img=output_g, channel='y')
        bin_mask1_2_c_g = pcv.threshold.otsu(gray_img=a_gray2_c_g, object_type='dark')
        bin_mask1_2_invert_g = pcv.invert(bin_mask1_2_c_g)
        clean_mask1_2_g = pcv.closing(gray_img=bin_mask1_2_invert_g, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2_g = pcv.fill(clean_mask1_2_g, 300)
        clean_mask1_21_g = pcv.opening(clean_mask1_2_g)

        a_gray2_l_g = pcv.rgb2gray_lab(rgb_img=output_g, channel='a')
        bin_mask1_2_g = pcv.threshold.otsu(gray_img=a_gray2_l_g, object_type='dark')
        #cv2.imshow("Threshold", bin_mask1_2)
        #cv2.waitKey(0)
        clean_mask1_2_g = pcv.closing(gray_img=bin_mask1_2_g, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2_g = pcv.fill(clean_mask1_2_g, 300)
        clean_mask1_21_g = pcv.opening(clean_mask1_2_g)


        size_c = np.count_nonzero(bin_mask1_2_invert)
        size_l = np.count_nonzero(bin_mask1_2)
        size_c_g = np.count_nonzero(bin_mask1_2_invert_g)
        size_l_g = np.count_nonzero(bin_mask1_2_g)

        print(np.count_nonzero(a_gray2_c_g))

        cmyk_output = cv2.putText(a_gray2_c, f'CMYK \"Y\" \\G', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        lab_output = cv2.putText(a_gray2_l, f'LAB \"A\" \\G', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        cmyk_bin_output = cv2.putText(bin_mask1_2_invert, f'CMYK Size {size_c}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        lab_bin_output = cv2.putText(bin_mask1_2, f'LAB Size {size_l}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        cmyk_bin_output_g = cv2.putText(bin_mask1_2_invert_g, f'CMYK Gamme Size {size_c_g}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        lab_bin_output_g = cv2.putText(bin_mask1_2_g, f'LAB Gamma Size {size_l_g}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        

        hori_colorspace = np.concatenate((lab_output, cmyk_output), axis=1)
        hori_colorspace_g = np.concatenate((hori_colorspace, cmyk_bin_output_g), axis=1)
        hori_bin = np.concatenate((lab_bin_output, cmyk_bin_output), axis=1)
        hori_bin_g = np.concatenate((hori_bin, lab_bin_output_g), axis=1)
        fin_im = np.concatenate((hori_colorspace_g, hori_bin_g), axis=0)
        cv2.imshow(None, fin_im)
        cv2.waitKey(0)





        a_gray2 = pcv.rgb2gray_cmyk(rgb_img=output_g, channel='y')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_25 = pcv.opening(clean_mask1_2)


        '''a_gray2 = pcv.rgb2gray_lab(rgb_img=img, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_2 = pcv.opening(clean_mask1_2)

        a_gray2 = pcv.rgb2gray_lab(rgb_img=output, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_21 = pcv.opening(clean_mask1_2)

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected_img1, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_22 = pcv.opening(clean_mask1_2)

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected_img2, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_23 = pcv.opening(clean_mask1_2)'''

        a_gray2 = pcv.rgb2gray_lab(rgb_img=img_g, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_24 = pcv.opening(clean_mask1_2)
        '''
        a_gray2 = pcv.rgb2gray_lab(rgb_img=output_g, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_25 = pcv.opening(clean_mask1_2)

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected1_g, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_26 = pcv.opening(clean_mask1_2)

        a_gray2 = pcv.rgb2gray_lab(rgb_img=corrected2_g, channel='a')
        bin_mask1_2 = pcv.threshold.otsu(gray_img=a_gray2, object_type='dark')
        clean_mask1_2 = pcv.closing(gray_img=bin_mask1_2, kernel=np.array([[0, 1, 1, 0],[1, 1, 1, 1],[1, 1, 1, 1],[0, 1, 1, 0]]))
        #fclean_mask1_2 = pcv.erode(clean_mask1_2, ksize=3, i=1)
        clean_mask1_2 = pcv.fill(clean_mask1_2, 300)
        clean_mask1_27 = pcv.opening(clean_mask1_2)
'''

        #hist_figure, hist_data = pcv.visualize.histogram(img=output_g, mask=clean_mask1_25, hist_data=True)

        #pcv.params.sample_label = "plant"
        

        #euclidean_lengths = pcv.outputs.observations['plant']['segment_eu_length']['value']
        #print(euclidean_lengths)

        shape_img = pcv.analyze.size(img, clean_mask1_2, 1)
        shape_out = pcv.analyze.size(output, clean_mask1_21, 1)
        #shape_cor1 = pcv.analyze.size(corrected_img1, clean_mask1_22, 1)
        #shape_cor2 = pcv.analyze.size(corrected_img2, clean_mask1_23, 1)

        shape_img_g = pcv.analyze.size(img_g, clean_mask1_24, 1)
        shape_out_g = pcv.analyze.size(output_g, clean_mask1_25, 1)
        #shape_cor1_g = pcv.analyze.size(corrected1_g, clean_mask1_26, 1)
        #shape_cor2_g = pcv.analyze.size(corrected2_g, clean_mask1_27, 1)

        size_1 = np.count_nonzero(clean_mask1_2)
        size_2 = np.count_nonzero(clean_mask1_21)
        #size_3 = np.count_nonzero(clean_mask1_22)
        #size_4 = np.count_nonzero(clean_mask1_23)
        #size_5 = np.count_nonzero(clean_mask1_24)
        size_6 = np.count_nonzero(clean_mask1_25)
        #size_7 = np.count_nonzero(clean_mask1_26)
        #size_8 = np.count_nonzero(clean_mask1_27)

        ov1 = pcv.visualize.overlay_two_imgs(img, clean_mask1_2, alpha=0.7)
        ov2 = pcv.visualize.overlay_two_imgs(img, clean_mask1_21, alpha=0.7)
        #ov3 = pcv.visualize.overlay_two_imgs(img, clean_mask1_22, alpha=0.7)
        #ov4 = pcv.visualize.overlay_two_imgs(img, clean_mask1_23, alpha=0.7)
        #ov5 = pcv.visualize.overlay_two_imgs(img, clean_mask1_24, alpha=0.7)
        ov6 = pcv.visualize.overlay_two_imgs(img, clean_mask1_25, alpha=0.7)
        #ov7 = pcv.visualize.overlay_two_imgs(img, clean_mask1_26, alpha=0.7)
        #ov8 = pcv.visualize.overlay_two_imgs(img, clean_mask1_27, alpha=0.7)





    
        
        # Using cv2.putText() method
        #bin1 = cv2.putText(ov1, f'Orig Size{size_1}', org, font, 
        #                fontScale, color, thickness, cv2.LINE_AA)
        bin2 = cv2.putText(ov2, f'Out Size{size_2}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        '''        bin3 = cv2.putText(ov3, f'WCx1 Size{size_3}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        bin4 = cv2.putText(ov4, f'WCx2 Size{size_4}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        bin5 = cv2.putText(ov5, f'Orig+GC Size{size_5}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)'''
        bin6 = cv2.putText(ov6, f'Out+GC Size{size_6}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        '''bin7 = cv2.putText(ov7, f'WCx1+GC Size{size_7}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        bin8 = cv2.putText(ov8, f'WCx2+GC Size{size_8}', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)'''
        
        color_scatter, _ = pcv.visualize.pixel_scatter_plot(paths_to_imgs = [img_path] , x_channel = "a", y_channel ="b" )

        hori1 = np.concatenate((img, output), axis=1)
        hori2 = np.concatenate((corrected_img1, corrected_img2), axis=1)
        hori3 = np.concatenate((hori1, hori2), axis=0)
        hori4 = np.concatenate((img_g, output_g), axis=1)
        hori5 = np.concatenate((corrected1_g, corrected2_g), axis=1)
        hori6 = np.concatenate((hori4, hori5), axis=0)

        cv2.imshow(None, hori3)
        cv2.waitKey(0)

        cv2.imshow(None, hori6)
        cv2.waitKey(0)

        hori1 = np.concatenate((shape_img, shape_out), axis=1)
        #hori2 = np.concatenate((shape_cor1, shape_cor2), axis=1)
        #hori3 = np.concatenate((hori1, hori2), axis=0)
        hori4 = np.concatenate((shape_img_g, shape_out_g), axis=1)
        #hori5 = np.concatenate((shape_cor1_g, shape_cor2_g), axis=1)
        #hori6 = np.concatenate((hori4, hori5), axis=0)


        cv2.imshow(None, hori1)
        cv2.waitKey(0)

        cv2.imshow(None, hori4)
        cv2.waitKey(0)


        #hori1 = np.concatenate((bin1, bin2), axis=1)
        #hori2 = np.concatenate((bin3, bin4), axis=1)
        #hori3 = np.concatenate((hori1, hori2), axis=0)
        #hori4 = np.concatenate((bin5, bin6), axis=1)
        #hori5 = np.concatenate((bin7, bin8), axis=1)
        #hori6 = np.concatenate((hori4, hori5), axis=0)
        hori_y_range = np.concatenate((bin2, bin6), axis=1)

        #cv2.imshow(None, hori3)
        #cv2.waitKey(0)

        #cv2.imshow(None, hori6)
        #cv2.waitKey(0)

        cv2.imshow(None, hori_y_range)
        cv2.waitKey(0)

        pcv.params.line_thickness = 5
        colorspace_img = pcv.visualize.colorspaces(rgb_img=output_g)
        cv2.imshow(None, colorspace_img)
        cv2.waitKey(0)

        
         # Convert the Matplotlib Figure (color_scatter) to a NumPy array for OpenCV display
        if isinstance(color_scatter, plt.Figure):
            color_scatter.canvas.draw()
            rgba_image = np.array(color_scatter.canvas.renderer.buffer_rgba())
            # Convert RGBA to BGR and remove alpha channel for OpenCV
            color_scatter_np = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGR)
            cv2.imwrite('pixel_scatter_plot.jpg', color_scatter_np)
            cv2.imshow("Pixel Scatter Plot", color_scatter_np)
            cv2.waitKey(0)
            plt.close(color_scatter) # Close the Matplotlib figure
        else:
            print("Warning: color_scatter was not a Matplotlib Figure.")

        

        cv2.destroyAllWindows()




ExtractLeafTips(folder_path)
