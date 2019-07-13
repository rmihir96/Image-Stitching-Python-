
"""
@author : Mihir Ranade
CVIP Project 2
UBIT : mihirraj
#no : 50290849
"""
import os
import argparse
import cv2
import random
import numpy as np
#from scipy import linalg as LA
#from scipy.spatial import distance




def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 2.")
    parser.add_argument(
        "dir_path", type=str, default="",
        help="path to the images used for  (do not change this arg)")
    args = parser.parse_args()
    return args



def find_matches(img1, img2):
    """
    args:
    Img1 & Img2: Images for which the features are to be matched

    returns:
    Points : Keypoints detected in image1 and image2
    """
    
    
    # Initiate ORB detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    #print(len(des1), len(des2))
    final_list = []
    trainIdx = []
    queryIdx = []

    for i, qdes in enumerate(des1):
        templist = []
        for j, tdes in enumerate(des2):
            norm_dist = cv2.norm(qdes, tdes, cv2.NORM_HAMMING)
            templist.append([norm_dist, j , i])
        sorted_temp = sorted(templist)
        final_list.append(sorted(templist)[:2]) 

    verify_match = []
    vm_trainIdx = []
    vm_queryIdx = []
    for i in range(len(final_list)):
        #print(final_list[i][0], final_list[i][1])
        if final_list[i][0][0] < 0.60 * final_list[i][1][0]:
            verify_match.append(final_list[i][0][0])
            vm_trainIdx.append(final_list[i][0][1])
            vm_queryIdx.append(final_list[i][0][2])
    #print(verify_match)



    # Mimnum number of matches
    min_matches = 8
    if len(verify_match) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for i in range(len(verify_match)):
            #print(match.trainIdx)
            img1_pts.append(kp1[vm_queryIdx[i]].pt)
            img2_pts.append(kp2[vm_trainIdx[i]].pt)
        img1_pts = np.float32(img1_pts).reshape(-1,1,2)
        img2_pts = np.float32(img2_pts).reshape(-1,1,2)

    else:
        print('Error: Not enough matches, please change the order')
        exit()

    return (img1_pts, img2_pts)

#Reference:- https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html


def find_homography(src_pts, dst_pts):
    """
    args:
    src_pts : Image1 key points
    dst_pts : Image2 key points

    returns:
    H : Homography matrix build for the set of points using DLT method
    """
    
    temp_A = []
    for i, j in zip(src_pts, dst_pts):
        p1 = np.matrix([i.item(0), i.item(1), 1])
        p2 = np.matrix([j.item(0), j.item(1), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        temp_A.append(a1)
        temp_A.append(a2)

    A = np.matrix(temp_A)
  
    
    S,U,V = np.linalg.svd(A)
    #print(V.shape)
    #print(V)
    
    #print(Vp)
    H = np.reshape(V[8],(3,3))
    

    H  = (1/H.item(8)) * H
    #print(H)
    return H


def Ransac(src, des):
    """
    args : 
    src - keypoints of source image
    des - keypoints of destination image

    returns: Model that has the best number of inliers : inshort the best Homography matrix.
    """
    src = src.reshape(src.shape[0],2)
    des = des.reshape(des.shape[0],2)
   
    max_H = None
    max_inliers = []

    for _ in range(5000):
        #print("the loop number %d" % _)
        #Randomly sample 4 points from source image
        random_points = src[np.random.choice(src.shape[0], 4, replace=False), :]
        #print(random_points)
        #print(random_points.shape)

        #Get indexes of these 4 random points.
        des_points = []
        pts_id = []
        for point in random_points:
            pts_id = (np.where(src == point))
            if len(pts_id) > 0 and len(pts_id[0]) > 0:
                des_points.append(des[pts_id[0][0]])
       # print(np.asarray(des_points).shape)
        #print(len(pts_id))
        #print(np.asarray(des_points).shape)

        src_points = random_points.reshape(4,1,2)
        des_points = np.asarray(des_points).reshape(4,1,2)

        #Use random_points and des_points to compute a 8X9 matrix which is used to find Homography.
        H = find_homography(src_points, des_points)


        inliers = []
        
        for i in range(len(src)):
            d = calculate_dist(src[i], des[i], H)
            if d < 4.0:
                inliers.append((src[i], des[i]))
            
            if len(inliers)> len(max_inliers):
                max_inliers = inliers
                max_H = H
            
#             if len(max_inliers) > (len(corr)*0.7):
#                 break
    #print(max_H)
       # print(max_inliers)
    return max_H, max_inliers
        

def calculate_dist(src, des, H):
    """
    args:
    src - Image1 points
    des - Image2 Points
    H : Homography matrix

    Returns:
    distance - Distance between predicted and actual image2
    """
    
    
    source_points = np.transpose(np.matrix([src.item(0), src.item(1), 1]))
    pred_dest = np.dot(H, source_points)
    pred_dest = (1/pred_dest.item(2))*pred_dest

    actual_dest = np.transpose(np.matrix([des.item(0), des.item(1), 1]))
    error = actual_dest - pred_dest
    distance =  np.linalg.norm(error)
    #print(distance)
    
    return distance
    
#Reference : https://github.com/hughesj919/HomographyEstimation/blob/master/Homography.py
    

# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):
    """
    args:
    img1 : Input image 1
    img2: Input image 2
    M : Homography matrix

    returns:
    Stitched_img : Stitched panaroma of image 1 and image 2.
    """

    # Get width and height of input images	
    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation 
    transform_dist = [-x_min,-y_min]
    transform_array = np.array([[1, 0, transform_dist[0]], 
                                [0, 1, transform_dist[1]], 
                                [0,0,1]]) 

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M), 
                                    (x_max-x_min, y_max-y_min))
    result_img[transform_dist[1]:w1+transform_dist[1], 
                transform_dist[0]:h1+transform_dist[0]] = img1

    # Return the result
    return result_img

#Reference:- https://github.com/pavanpn/Image-Stitching/blob/master/stitch_images.py


def create_panaroma(img_, img):

    img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #print(img1.shape)
    #print(img2.shape)

    src, des = find_matches(img1,img2)

    H, inliers = Ransac(src, des)


    # # Stitch the images together using homography matrix
    stitched_image = get_stitched_image(img, img_, H)

    return stitched_image


# # Write the result to the same directory
# #result_image_name = 'results/result_'+sys.argv[1]


def main():
    args = parse_args()
    print("loading images...")
    imagePaths = (os.listdir(args.dir_path))
    print(imagePaths)
    images = []
    for imagePath in imagePaths:
        image = cv2.imread(os.path.join(args.dir_path, imagePath))
        if image is not None:
            images.append(image)
    #print(images)

    panorama = images[0]
    for i in range(1,len(images)):
        #print(images[i])
        panorama = create_panaroma(panorama,images[i])

    print("Image Stitching completed.")
    cv2.imwrite(os.path.join(args.dir_path , 'panaroma.jpg'), panorama)
    #cv2.imwrite("output.jpg", panorama)



if __name__ == "__main__":
    
    main()
    