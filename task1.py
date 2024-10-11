import cv2
import numpy as np

car_image = cv2.imread('assignment/images/1.jpeg', cv2.IMREAD_COLOR)
car_image = car_image.astype(np.float32)
car_image = car_image / 255.0
car_mask = cv2.imread('assignment/car_masks/1.png', cv2.IMREAD_COLOR)
car_mask = car_mask.astype(np.float32)
car_mask = car_mask / 255.0
floor = cv2.imread('assignment/floor.png', cv2.IMREAD_COLOR)
wall = cv2.imread('assignment/wall.png', cv2.IMREAD_COLOR)
shadow = cv2.imread('assignment/shadow_masks/1.png',  cv2.IMREAD_COLOR)
shadow = shadow.astype(np.float32)
shadow = shadow / 255.0


portion_of_wall = wall[600:1600, :, :]
portion_of_floor = floor[1430:, :, :]
combined_image = np.vstack((portion_of_wall, portion_of_floor))
combined_image = combined_image[:, 210:3670, :]  # Crop to desired region
desired_size = (1920, 1080)  # Resize to desired size
combined_image = cv2.resize(combined_image, desired_size)
combined_image = combined_image.astype(np.float32)
combined_image = combined_image/ 255.0
combined_image_p = np.clip((combined_image/np.max(combined_image))*255, 0, 255).astype(np.uint8)
cv2.imwrite('combined_background.png', combined_image_p)
fil_car_mask = cv2.medianBlur(car_mask, ksize=3)
fil_car_mask_p = np.clip((fil_car_mask/np.max(fil_car_mask))*255, 0, 255).astype(np.uint8)
cv2.imwrite('median_filter.png', fil_car_mask_p)


n=10 #img1
# n =  5 #img2
# n=5 #img3
# n = 6 #img4  
# n = 10 #img 5
# n = 10 #img 6


X, Y = combined_image.shape[1], combined_image.shape[0]
roi = combined_image[int(Y/n):, int(X/n):int((n-1)*X/n)]

# car = cv2.bitwise_and(car_image, fil_car_mask)
# cv2.imwrite('car_only.png', car)


# mask_inv = cv2.bitwise_not(fil_car_mask)
# cv2.imwrite('mask_imv.png', mask_inv)
# mask_inv_resized = cv2.resize(mask_inv, (roi.shape[1], roi.shape[0]))
# back_no_car = cv2.bitwise_and(roi, mask_inv_resized)
# cv2.imwrite('back_no_car.png', back_no_car)
# car_resized = cv2.resize(car, (roi.shape[1], roi.shape[0]))
# car_back = back_no_car + car_resized
# cv2.imwrite('back_with_car.png', car_back)

alpha = fil_car_mask
alpha_resized = cv2.resize(alpha, (roi.shape[1], roi.shape[0]))
car_resized = cv2.resize(car_image, (roi.shape[1], roi.shape[0]))
car_back = cv2.multiply(alpha_resized, car_resized) + cv2.multiply(1 - alpha_resized, roi)
car_back_p = np.clip((car_back/np.max(car_back))*255, 0, 255).astype(np.uint8)
cv2.imwrite('background_roi_with_car.png', car_back_p)
result = combined_image.copy()
result[int(Y/n):, int(X/n):int((n-1)*X/n)] = car_back
result_p = np.clip((result/np.max(result))*255, 0, 255).astype(np.uint8)
cv2.imwrite('combined_image_with_car_without_shadow.png', result_p)



#shadoww
scale_x  = float(((n-2)*combined_image.shape[1]/(n))/car_image.shape[1])
scale_y  = float(((n-2)*combined_image.shape[0]/n)/car_image.shape[0])

tpx, tpy = (car_image.shape[1]-shadow.shape[1])/2 , car_image.shape[0]-shadow.shape[0]


adjust_y = 10 ##for img1
adjust_x = 38



# adjust_y = 6 ##for img2
# adjust_x = 25

# adjust_y = 85 ##for img3
# adjust_x = 0

# adjust_y = -145##for img4
# adjust_x = 32

# adjust_y = -155#for img5
# adjust_x = 50

# adjust_y = 60#for img6
# adjust_x = 50
ntpx, ntpy = int(tpx*scale_x), int(tpy*scale_y) 
nsizex, nsizey = int(scale_x*shadow.shape[1]), int(scale_y*shadow.shape[0])

ntpx_cm , ntpy_cm = int(combined_image.shape[1]/n + ntpx), int(combined_image.shape[0]/n + ntpy)
roi  = result[ ntpy_cm+adjust_y:ntpy_cm+nsizey+adjust_y, ntpx_cm+adjust_x:ntpx_cm+nsizex+adjust_x]
roi_p = np.clip((roi/np.max(roi))*255, 0, 255).astype(np.uint8)
cv2.imwrite('roi.png', roi_p)
shadow_resized = cv2.resize(shadow, (roi.shape[1], roi.shape[0]))
shadow_resized_p = np.clip((shadow_resized/np.max(shadow_resized))*255, 0, 255).astype(np.uint8)

cv2.imwrite('shadow_resized.png', shadow_resized_p)

roi_shadow = result[ntpy_cm + adjust_y:ntpy_cm + nsizey + adjust_y, ntpx_cm + adjust_x:ntpx_cm + nsizex + adjust_x]
roi_shadow_p = np.clip((roi_shadow /np.max(roi_shadow ))*255, 0, 255).astype(np.uint8)
cv2.imwrite('roi_shadow.png', roi_shadow_p)

# Resize
shadow_mask_resized = cv2.resize(shadow, (roi_shadow.shape[1], roi_shadow.shape[0]))
shadow_mask_blurred = cv2.GaussianBlur(shadow_mask_resized, (21, 21), 0)
shadow_mask_blurred_p = np.clip((shadow_mask_blurred /np.max(shadow_mask_blurred ))*255, 0, 255).astype(np.uint8)
cv2.imwrite('shadow_mask_blurred.png', shadow_mask_blurred_p)


#shadow to the ROI
shadow_intensity = 0.5
shadow_layer = roi_shadow.copy().astype(np.float32)
shadow_layer[:, :, 0] *= (1 - shadow_intensity * shadow_mask_blurred[:, :, 0])
shadow_layer[:, :, 1] *= (1 - shadow_intensity * shadow_mask_blurred[:, :, 1])
shadow_layer[:, :, 2] *= (1 - shadow_intensity * shadow_mask_blurred[:, :, 2])
shadow_layer_p = np.clip((shadow_layer /np.max(shadow_layer))*255, 0, 255).astype(np.uint8)
cv2.imwrite('shadow_layer.png', shadow_layer_p)

result_with_shadow = result.copy()
result_with_shadow[ntpy_cm + adjust_y:ntpy_cm + nsizey + adjust_y, ntpx_cm + adjust_x:ntpx_cm + nsizex + adjust_x] = shadow_layer

final_result = cv2.GaussianBlur(result_with_shadow, (3, 3), 0)
final_result_p = np.clip((final_result /np.max(final_result))*255, 0, 255).astype(np.uint8)
cv2.imwrite('1_final_with_shadow.png', final_result_p)













