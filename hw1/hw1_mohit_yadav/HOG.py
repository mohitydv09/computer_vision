import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def get_differential_filter():
    # To do
    # Using Prewitt Filter.
    filter_x = np.array([[1,0,-1],
                         [1,0,-1],
                         [1,0,-1]])
    filter_y = np.array([[1,1,1],
                         [0,0,0],
                         [-1,-1,-1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    ## Pad the image according to the filter size.
    # Get filter size
    k = filter.shape[0]    # Assumed that the pad will always be odd.
    pad_size = (k-1)//2

    # Placeholder padded image.
    im_padded = np.zeros((im.shape[0] + 2*pad_size, im.shape[1] + 2*pad_size))
    im_padded[1:-1,1:-1] = im
    
    m, n = im.shape
    im_filtered = np.zeros((m,n))    #Placeholder for filteered Image
    # loop to go to every pixel calculate its value and store in im_filtered.
    for i in range(pad_size, m+pad_size):
        for j in range(pad_size, n+pad_size):
            im_block = im_padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            # Applying correlation here.
            im_filtered[i-pad_size,j-pad_size] = np.sum(im_block * filter)
    # print("Filter Image ran, shape of filtered Image is : ",im_filtered.shape)
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    # Go to pixels in the image and replace with the magnitute.
    m,n = im_dx.shape
    # Make placeholder np arrays.
    grad_mag = np.zeros((m,n))
    grad_angle = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            grad_mag[i,j] = np.sqrt(np.square(im_dx[i,j]) + np.square(im_dy[i,j]))
            angle = np.arctan2(im_dy[i,j], im_dx[i,j])
            # To make the angle as given in problem i.e. between 0 to Pi.
            if (angle < 0):
                angle += np.pi
            grad_angle[i,j] = angle
    # print("Grad Mag formed with shape : ", grad_mag.shape)
    # print("Grad Angle formed with shape : ", grad_angle.shape)
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    # Get the size of histogram that will be formed, enges that are not in this size will be truncated.
    M = grad_mag.shape[0]//cell_size
    N = grad_mag.shape[1]//cell_size
    # Create placeholder histogram, here depht of 6 is given manually as per problem.
    ori_histo = np.zeros((M,N,6))

    # Convert the angels to degree for easy calculations.
    grad_angle = np.degrees(grad_angle)

    # Iterate over all the cells.
    for i_cell in range(M):
        for j_cell in range(N):
            # Iterate insde the cells.
            for i in range(i_cell*cell_size, i_cell*cell_size + cell_size):
                for j in range(j_cell*cell_size, j_cell*cell_size + cell_size):
                    # get the bin in which this angle has to be placed.
                    bin_num = int((grad_angle[i,j] + 15)//30)
                    if bin_num == 6:
                        bin_num = 0
                    # Do magnitude addition.
                    ori_histo[i_cell,j_cell,bin_num] += grad_mag[i,j]
    # print("Ori history formed with size : ", ori_histo.shape)
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    M,N,d = ori_histo.shape
    ori_histo_normalized = np.zeros((M - block_size + 1, N - block_size + 1, d*block_size*block_size))

    # Iterate over cells.
    for i in range(M-block_size+1):
        for j in range(N-block_size+1):
            # Make a concatenated array.
            concat_array = []
            # Iterate over block size.
            for m in range(block_size):
                for n in range(block_size):
                    # print('Curr Ori Hist : ', ori_histo[i+m, j+n,:])
                    concat_array += list(ori_histo[i+m,j+n,:])
            concat_array = np.array(concat_array)
            norm_of_array = np.linalg.norm(concat_array)
            concat_array = concat_array/(norm_of_array+0.000001)
            ori_histo_normalized[i,j,:] = concat_array 
    # print("Ori History Normalized and retured with shape : ",ori_histo_normalized.shape)
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0   # Converted to float and normalized.
    # To do
    # Get the diffrential filter that are to be used.
    filter_x, filter_y = get_differential_filter()
    filter_image_x = filter_image(im, filter_x)
    filter_image_y = filter_image(im, filter_y)

    grad_mag, grad_angle = get_gradient(filter_image_x, filter_image_y)

    cell_size = 8
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)

    block_size = 2
    hog = get_block_descriptor(ori_histo, block_size)

    # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    # To do
    M, N = I_target.shape
    m, n = I_template.shape
    # Initialize bounding box array.
    bounding_boxes = np.zeros((1,3))
    # print("Target Shape : ", I_target.shape)
    # print("Template Shape : ", I_template.shape)

    # Get the HOG.
    hog_template = extract_hog(I_template).flatten()
    # print("Shape of Flattened Template : ", hog_template.shape)

    # print("First Loop : ", M-m+1)
    # print("Second Loop : ", N-n+1)

    for i in range(0,M-m+1,3):
        # print("Current i is : ",i)
        for j in range(0,N-n+1,3):

            sub_image = I_target[i:i+m, j:j+n]

            hog_image = extract_hog(sub_image).flatten()
            # print("Shape of flattened image : ", hog_image.shape)

            # u_image = sub_image.reshape(-1)
            # v_temp = I_template.reshape(-1)
            image_norm = hog_image - np.mean(hog_image)
            template_norm = hog_template - np.mean(hog_template)
            # print("Shape of image : ", image_norm.shape )
            # print("Shape of Template : ", template_norm.shape)
            score = np.dot(image_norm, template_norm) / (np.linalg.norm(image_norm) * np.linalg.norm(template_norm))
            # Add all the items to bbs
            if (i==0) and (j==0):
                # First time running.
                # Note the i,j are reversed as per x,y specified in the problem.
                bounding_boxes[0] = np.array([j,i,score])
            else:
                bounding_boxes = np.vstack((bounding_boxes, np.array([j,i,score])))
    
    # print("Shape of BB after adding everything is : ", bounding_boxes.shape)

    # Thresholding.
    # Threshold was tuned manually.
    threshold = bounding_boxes[:,2]>0.48
    bounding_boxes = bounding_boxes[threshold]
    # print("Shape of BB after thres : ", bounding_boxes.shape)

    # Non-maximal Suppression.
    final_bb = np.zeros((1,3))
    while(len(bounding_boxes)>0):
        # Find the max bb.
        max_score = np.argmax(bounding_boxes[:,2])
        max_bb = bounding_boxes[max_score,:]
        # Delete max_bb form this set.
        final_bb = np.vstack((final_bb, max_bb))
        bounding_boxes = np.delete(bounding_boxes, max_score, axis=0)

        i = 0
        while i < len(bounding_boxes):
            # We have max_bb and bb here.
            # w is the difffrence between starting point of the two bbs in x direction.
            w = abs(max_bb[1] - bounding_boxes[i,1])
            h = abs(max_bb[0] - bounding_boxes[i,0])
            # print("w is : ", w ," h is : ", h)
            # if the image don't overlap
            if(w > m) or (h > n):
                i+=1
                continue
            # h = abs(max_bb[0] - bb[0])
            int_x = m - w     # Int means intersection here
            int_y = n - h    
            int_area = int_x*int_y
            iou = int_area / (2*m*n - int_area)
            # print("IoU is : ", iou)
            if iou > 0.5:
                # delete the item.
                bounding_boxes = np.delete(bounding_boxes, i, axis=0)
            else:
                i+=1
    #Remove the extra first row that we had.
    bounding_boxes = final_bb[1:,:]
    # print("Final items in bb are : ", len(bounding_boxes))
    return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)

    hog = extract_hog(im)

    I_target = cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # this is visualization code.