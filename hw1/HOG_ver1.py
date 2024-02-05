import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
Do not change the input/output of each function, and do not remove the provided functions.
'''

def get_differential_filter():
    # To do
    filter_x = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    filter_y = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    ## Pad the image according to the filter size.
    k = filter.shape[0]
    pad_size = k//2
    im_padded = np.zeros((im.shape[0] + 2*pad_size, im.shape[1] + 2*pad_size))
    im_padded[1:-1,1:-1] = im
    
    m, n = im.shape
    im_filtered = np.zeros((m,n))
    # loop to go to every pixel calculate its value and store in im_filtered.
    for i in range(pad_size, m+pad_size):
        for j in range(pad_size, n+pad_size):
            im_block = im_padded[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            im_filtered[i-pad_size,j-pad_size] = np.sum(im_block*filter)
    # im_filtered = im_filtered + 1
    # im_filtered = im_filtered*(255/2)
    # im_filtered = im_filtered.astype(np.uint8)
    # cv2.imshow("Image ", im_filtered)
    # cv2.waitKey(0)
    return im_filtered


def get_gradient(im_dx, im_dy):
    # To do
    # Go to pixels in the image and replace with the magnitute.
    m,n = im_dx.shape
    grad_mag = np.zeros((m,n))
    grad_angle = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            curr_mag = np.sqrt(np.square(im_dx[i,j]) + np.square(im_dy[i,j]))
            grad_mag[i,j] = curr_mag
            grad_angle[i,j] = (np.arctan(im_dy[i,j] / (im_dx[i,j] + 0.000001)) + np.pi/2)   # Epsilon added here to remove division by zero error.
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    M = grad_mag.shape[0]//cell_size
    N = grad_mag.shape[1]//cell_size
    ori_histo = np.zeros((M,N,6))

    # Convert the angels to degree for easy calculations.
    grad_angle = np.degrees(grad_angle)
    for i_cell in range(M):
        for j_cell in range(N):
            for i in range(i_cell*cell_size, i_cell*cell_size + cell_size):
                for j in range(j_cell*cell_size, j_cell*cell_size + cell_size):
                    bin_num = int((grad_angle[i,j] + 15)//30)
                    if bin_num == 6:
                        bin_num = 0
                    ori_histo[i_cell,j_cell,bin_num] += grad_mag[i,j]
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0   # Converted to float and normalized.
    # To do
    filter_x, filter_y = get_differential_filter()
    filter_image_x = filter_image(im, filter_x)
    filter_image_y = filter_image(im, filter_y)

    grad_mag, grad_angle = get_gradient(filter_image_x, filter_image_y)
    cell_size = 8
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)

    # visualize to verify
    # visualize_hog(im, hog, 8, 2)

    # return hog
    return 1


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

    # I_target = cv2.imread('target.png', 0)
    # #MxN image

    # I_template = cv2.imread('template.png', 0)
    # #mxn  face template

    # bounding_boxes=face_recognition(I_target, I_template)

    # I_target_c= cv2.imread('target.png')
    # # MxN image (just for visualization)
    # visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # #this is visualization code.
