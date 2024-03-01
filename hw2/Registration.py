import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


def visualize_align_image_using_feature(img1, img2, x1, x2, A, ransac_thr, img_h=500):
    x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ A.T
    errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
    mask_inliers = errors < ransac_thr
    boundary_t = np.hstack(( np.array([[0, 0], [img1.shape[1], 0], [img1.shape[1], img1.shape[0]], [0, img1.shape[0]], [0, 0]]), np.ones((5, 1)) )) @ A[:2, :].T

    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    boundary_t = boundary_t * scale_factor2
    boundary_t[:, 0] += img1_resized.shape[1]
    plt.plot(boundary_t[:, 0], boundary_t[:, 1], 'y')
    for i in range(x1.shape[0]):
        if mask_inliers[i]:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'r')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'ro')
    plt.axis('off')
    plt.show()


def find_match(img1, img2):
    # To do
    # Create a sift object.
    sift = cv2.SIFT_create()

    # Extract the keypoints and the corresponding descriptors.
    keypoints_1, descriptor_1 = sift.detectAndCompute(img1, None)
    keypoints_2, desciptior_2 = sift.detectAndCompute(img2, None)

    # Create a Nearest Neighbour object.
    neigh = NearestNeighbors(n_neighbors = 2) # We only need two points, second point is used for ratio test.
    
    # Fit the algo to the data that we have, we fit the second image and run the algo for points in img1.
    neigh.fit(desciptior_2)
    distances, indices = neigh.kneighbors(descriptor_1, n_neighbors = 2, return_distance=True )

    # Find the distance ratio between first and second value.
    distance_ratio = distances[:,0]/distances[:,1]

    # Get the indices of points which correspond to each other from the two images.
    indices_1 = np.where(distance_ratio < 0.7)[0]
    indices_2 = indices[np.where(distance_ratio < 0.7)][:,0]

    # Get the coordinates of points and store in x1, x2
    x1 = np.array([[keypoints_1[i].pt[0], keypoints_1[i].pt[1]] for i in indices_1])
    x2 = np.array([[keypoints_2[i].pt[0], keypoints_2[i].pt[1]] for i in indices_2])

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    np.random.seed(32) # Set random seed for debugging.
    
    max_inliner = 0
    # Go in a loop for number of iterations.
    for _ in range(ransac_iter):
        # Get index points of three points from a list of indices. 
        # We need three correspondance points for a Affine transformation.
        index_points = np.random.choice(len(x1), 3,replace=False) # Output will a numpy array.

        # Get the three points at the random indices we generated.
        u1, v1 = x1[index_points[0]][0], x1[index_points[0]][1]
        u2, v2 = x1[index_points[1]][0], x1[index_points[1]][1]
        u3, v3 = x1[index_points[2]][0], x1[index_points[2]][1]
        u1_, v1_ = x2[index_points[0]][0], x2[index_points[0]][1]
        u2_, v2_ = x2[index_points[1]][0], x2[index_points[1]][1]
        u3_, v3_ = x2[index_points[2]][0], x2[index_points[2]][1]

        # Make the M matrix in equation Mx = b
        M = np.array([[u1, v1, 1, 0, 0, 0],
                    [0, 0, 0, u1, v1, 1],
                    [u2, v2, 1, 0, 0, 0],
                    [0, 0, 0, u2, v2, 1],
                    [u3, v3, 1, 0, 0, 0],
                    [0, 0, 0, u3, v3, 1]])
        
        # Create a b matrix in equation Mx = b.
        b = np.array([  u1_,
                        v1_,
                        u2_,
                        v2_,
                        u3_,
                        v3_])
        
        # I used loops and implemented the squared distance for this but now replaced it with Prof's code for consistency.
        # Find the affine transform matrix (x matrix)
        x = np.matmul( np.matmul( np.linalg.inv( np.matmul(M.T, M)), M.T) , b)
        # Make the affine matrix from x.
        temp_A = np.array([[x[0], x[1], x[2]],
                           [x[3], x[4], x[5]],
                           [0,     0,    1]])
        
        # Find the predicted x2 points. 
        x2_t = np.hstack((x1, np.ones((x1.shape[0], 1)))) @ temp_A.T    # Transpose of equation is done.

        # Find the error in transformation for each point.
        # Sum of square diffrence error.
        errors = np.sum(np.square(x2_t[:, :2] - x2), axis=1)
        inliners = np.sum(errors < ransac_thr)   # find the number of inliners.

        if inliners > max_inliner:
            A = temp_A    # Set the return affine transfrom as this one
            max_inliner = inliners   # Set new value for the max_inliners.
    return A


def warp_image(img, A, output_size):
    # To do    
    # Make a matrix of all the coordinate points in wraped image.
    x_wraped, y_wraped = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))
    input_points = np.column_stack((x_wraped.ravel(), y_wraped.ravel(), np.ones(len(x_wraped.ravel()))))

    # Map all the input points to corresponding location in target. Last col of ones is removed.
    mapped_points = (input_points @ A.T)[:,0:2]

    # Make a tuple of grid points as needed by interpd funtion.
    grid = (np.arange(img.shape[1]), np.arange(img.shape[0]))   # Tuple of grid points.
    final_values = interpolate.interpn(grid, img.T, mapped_points, bounds_error=False, fill_value=0) # default linear interpolation is used.

    # Reshape to get the final image.
    img_warped = final_values.reshape(output_size)

    return img_warped


def align_image(template, target, A):
    # To do
    
    # Try Normalizing the images to 0-1.
    template = template/255
    target = target/255

    # Compute gradient of template.
    im_dx = cv2.Sobel(template, cv2.CV_64F, 1, 0, ksize = 3)  # This is in x direction i.e. u.
    im_dy = cv2.Sobel(template, cv2.CV_64F, 0, 1, ksize = 3)  # This is in y direction i.e. v.
    grad_tpl = np.stack([im_dx, im_dy], axis=2)

    # Compute Jacobian of image.
    jacob = np.zeros((template.shape[0], template.shape[1], 2, 6))
    row_indices, col_indices = np.indices(template.shape)  # this will give a row and a column array with indices values.

    jacob[:,:,0,0] = col_indices   # Column Indices will be broadcasted to map all items.
    jacob[:,:,0,1] = row_indices
    jacob[:,:,0,2] = 1
    jacob[:,:,1,3] = col_indices
    jacob[:,:,1,4] = row_indices
    jacob[:,:,1,5] = 1

    # Compute Steepest descent images.
    sd_images = np.zeros((template.shape[0], template.shape[1], 6))
    for i in range(6):
        sd_images[:,:,i] = (grad_tpl[:,:,0] * jacob[:,:,0,i]) + (grad_tpl[:,:,1] * jacob[:,:,1,i])

    # Compute Hessian.
    # Reshape the sd_images
    sd_images_reshaped = sd_images.reshape((template.shape[0], template.shape[1], 1, 6))
    sd_images_reshaped_T = np.transpose(sd_images_reshaped, (0,1,3,2))
    hessian_x = np.matmul(sd_images_reshaped_T, sd_images_reshaped)
    hessian = np.sum(hessian_x, axis=(0,1))
    
    # Initialize errors list.
    errors=[]

    iter_counter = 0
    iter_threshold = 5000  # Provided so that
    # Go in the optimization loop.
    while iter_counter<iter_threshold:
        # print("iter no : ", iter_counter)
        iter_counter+=1

        # Wrap the target image to template.
        image_target = warp_image(target, A, template.shape)

        # Get the error Image.
        image_error = image_target - template

        # Add the error to a list for visualization.
        errors.append(np.linalg.norm(image_error))

        # Compute F.
        F_x = sd_images * image_error.reshape((template.shape[0], template.shape[1], 1))
        F = np.sum(F_x, axis=(0,1)).reshape((6,1))

        # Compute del_p. 
        ### Step size of 5 is given here to converge faster
        del_p = (np.linalg.inv(hessian) @ F) * 5
    
        # Update the Affine Transform.
        A = A @ np.linalg.inv(np.array([[del_p[0][0]+1, del_p[1][0], del_p[2][0]],
                                        [del_p[3][0], del_p[4][0]+1, del_p[5][0]],
                                        [0          , 0          , 1          ]]))
        
        thresh = 0.01
        norm_del_p = np.linalg.norm(del_p)
        if norm_del_p < thresh:
            break

    # Set the final A matrix for returning.
    A_refined = A
    # Convert errors to a np array for visualization.
    errors = np.array(errors)
    return A_refined, errors


def track_multi_frames(template, img_list):
    # To do
    # Initialize A.
    x1, x2 = find_match(template, img_list[0])
    ransac_thr = 120
    ransac_iter = 500
    A = align_image_using_feature(x1,x2,ransac_thr, ransac_iter)
    
    A_list = []
    for img in img_list:
        A_refined, _ = align_image(template, img, A)
        A_list.append(A_refined)
        A = A_refined
        template = warp_image(img, A, template.shape)

    return A_list

if __name__ == '__main__':
    template = cv2.imread('./JS_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./JS_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)
    
    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 120   # This was tuned manually to get ~70 percent inliners.
    ransac_iter = 500   # This was tuned using manually to get good result consistently, 
                        #probablistic analysis is giving a very low value.
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_align_image_using_feature(template, target_list[0], x1, x2, A, ransac_thr, img_h=500)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='grey', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    plot_error_map = False
    if plot_error_map:
        error_map = np.abs(img_warped - template)
        plt.imshow(error_map, cmap='jet', vmin=0, vmax=255)
        plt.axis('off')
        plt.show()

    A_refined, errors = align_image(template, target_list[1], A)   # Update from Prof.
    visualize_align_image(template, target_list[1], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)





