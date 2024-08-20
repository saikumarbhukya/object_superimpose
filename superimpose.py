import os
import cv2
import numpy as np
import random
import imutils
import math
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (20,20)
from PIL import Image
from matplotlib.pyplot import imshow
import json
import glob
import secrets
import sys
from torchvision import transforms
from torchvision.transforms import InterpolationMode
random.seed(123457)
####Read the source Image####
### Read the source Image Randomly ###
#transform = transforms.Compose([transforms.ToPILImage(),
                                #transforms.ColorJitter(brightness=[1,2],contrast=[1,2])])#,hue=[0,0.5],saturation=[1,2])])
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.ColorJitter(brightness=[1,2],contrast=[1,2],hue=[-0.5,0.5],saturation=[1,2])])
#transform = transforms.Compose([transforms.ToPILImage()])

class SuperImpose:
    def __init__(self):
        self.object_source = "/home/zestiot/Downloads/water_mark1/new2/*.jpeg"#"/home/suchith/coffee/*.jpg"
        self.target_source = "/home/zestiot/Downloads/Empty_Belt_1/Empty_Belt/*.jpg"
        self.source_image = glob.glob(self.object_source)
        self.target_image = glob.glob(self.target_source)
        #self.angle = [i for i in range(10, 100, 10)]
        self.angle = [10, 20, 40, 60, 180, 120]
        self.img_name = 5900
        self.i = 0


    def source(self):
        try:
            source_image = random.choice(self.source_image)
            source_image_ = source_image
            source_image_name = source_image
            source_image = transform(cv2.imread(source_image))
            source_image = np.array(source_image)[:, :, :].copy()
            source_image_json = source_image_name.split('.')[0] + ".json"
            return source_image, source_image_json, source_image_
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(
                f" Exception occurred in Source :", 'error')
            print(
                f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')


    def target(self):
        target_image = random.choice(self.target_image)
        target_image = cv2.imread(target_image)
        return target_image

    def dummy_source(self,source_img):
        h, w, _ = source_img.shape
        dummy_image_source = (np.zeros((h, w, 3), dtype=np.uint8))
        return dummy_image_source

    ### Define Dummy Image ###
    def dummy(self):
        dummy_image = (np.zeros((390, 2048, 3), dtype=np.uint8))
        return dummy_image


    ### Define the Mask ###
    def poly_fill_object(self,dummy_img, points_array):
        dummy_image_poly_fill = cv2.fillPoly(dummy_img, pts=[np.int32(points_array)], color=(255, 255, 255))
        return dummy_image_poly_fill


    ### Extract Object ###
    def extract_target_image(self,source_img, mask, inv):
        if inv == False:
            target_masked = cv2.bitwise_and(source_img, source_img, mask=mask[:, :, 1])
        elif inv == True:
            mask = cv2.bitwise_not(mask)
            target_masked = cv2.bitwise_and(source_img, source_img, mask=mask[:, :, 1])
        return target_masked


    ### Find the Bounding Box of the Objects ###
    def bounding_box_object(self,dummy_image_poly):
        try:
            gray = cv2.cvtColor(dummy_image_poly, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            bounding_box = cv2.boundingRect(contours[0])
            desired_x_max = bounding_box[0] + bounding_box[2]
            desired_y_max = bounding_box[1] + bounding_box[3]
            p,q = [bounding_box[0],bounding_box[1]],[desired_x_max,desired_y_max]
            diag = np.linalg.norm(np.array(p) - np.array(q))
            print(diag)
            if diag > 20:
                return bounding_box
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(
                f" Exception occurred in bounding box object :", 'error')
            print(
                f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')

    ### Extract Object Boundaries and place in target dummy image ###
    def obj_boundary(self,extracted_object, bounding_box, desired_y, desired_x):
        try:
            dummy_image_1 = self.dummy()
            object_region = extracted_object[bounding_box[1]:bounding_box[1] + bounding_box[3],
                            bounding_box[0]:bounding_box[0] + bounding_box[2]]
            dummy_image_1[desired_y:desired_y + object_region.shape[0],
            desired_x:desired_x + object_region.shape[1]] = object_region
            return dummy_image_1
        except Exception as e_:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(
                f" Exception occurred in obj_boundary:", 'error')
            print(
                f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')


    def make_target_img_mask(self,target_image, dummy_image_1):
        gray_1 = cv2.cvtColor(dummy_image_1, cv2.COLOR_BGR2GRAY)
        ret, thresh_1 = cv2.threshold(gray_1, 2, 255, cv2.THRESH_BINARY)
        gray2rgb_1 = cv2.cvtColor(thresh_1, cv2.COLOR_GRAY2RGB)
        mask_inv_1 = cv2.bitwise_not(gray2rgb_1)
        target_masked = cv2.bitwise_and(target_image, target_image, mask=mask_inv_1[:, :, 1])
        return target_masked


    def obtain_valid_positions(self,width_object, height_object):
        try:
            while True:
                desired_x, desired_y = random.randint(250, 1750), random.randint(0, 300)
                if (desired_x + width_object) < 1800 and (desired_y + height_object) < 390:
                    desired_x_max = desired_x + width_object
                    desired_y_max = desired_y + height_object
                    x_center = float((desired_x + desired_x_max)) / 2 / 2048
                    y_center = float((desired_y + desired_y_max)) / 2 / 390
                    wid = float(width_object) / 2048
                    hei = float(height_object) / 390
                    print("Obtaining Valid Positions")
                    return desired_x, desired_y, [x_center, y_center, wid, hei]


        except Exception as e_:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(
                f" Exception occurred in obtain valid positions :", 'error')
            print(
                f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')


    def read_json_points(self,json_file_name):
        with open(json_file_name) as user_file:
            file_contents = json.load(user_file)
        temp_list = []
        for num in range(len(file_contents["shapes"])):
            temp_list.append(file_contents["shapes"][num]["points"])
        return temp_list

    def main(self):
        while self.i < 100:
            try:
                source_img_, source_image_json, source_image_name = self.source()
                dummy_img_ = self.dummy_source(source_img_)
                points_array = self.read_json_points(source_image_json)
                for num in range(len(points_array)):
                    poly_fill_object_ = self.poly_fill_object(dummy_img_, points_array[num])
                    count = 0
                    extracted_object = self.extract_target_image(source_img_, poly_fill_object_, False)
                    extracted_object_bounding_box_1 = self.bounding_box_object(extracted_object)
                    if extracted_object_bounding_box_1:

                        while True:
                            #extracted_object = self.extract_target_image(source_img_, poly_fill_object_, False)
                            extracted_object = imutils.rotate(extracted_object, random.choice(self.angle))
                            extracted_object_bounding_box = self.bounding_box_object(extracted_object)
                            if extracted_object_bounding_box:

                                _, __, width_object, height_object = extracted_object_bounding_box
                                desired_x, desired_y, yolo_coordinates = self.obtain_valid_positions(width_object,
                                                                                                height_object)
                                target_image_position_mask = self.obj_boundary(extracted_object,
                                                                          extracted_object_bounding_box, desired_y,
                                                                          desired_x)
                                target_img_ = self.target()
                                target_image_to_add = self.make_target_img_mask(target_img_, target_image_position_mask)
                                synthetic_result_img = cv2.add(target_image_to_add, target_image_position_mask)
                                cv2.imwrite("/home/zestiot/Downloads/water_mark1/water_mark_sub_imposed/" + str(self.img_name) + ".jpg",
                                            synthetic_result_img)
                                with open("/home/zestiot/Downloads/water_mark1/water_mark_sub_imposed/" + str(self.img_name) + ".txt",
                                          'w') as f:
                                    f.write(f"1 {yolo_coordinates[0]} {yolo_coordinates[1]} {yolo_coordinates[2]} {yolo_coordinates[3]}")
                                count = count + 1
                                self.img_name += 1
                                #return synthetic_result_img
                                print(f"IM inside nested While loop and count is {count}")
                                if count == 3:
                                    break
                            else:
                                print(f"No Extracted bounding Boxes")
                                break
                    print("IM inside for loop of points")
                self.i = self.i + 1
                print(self.i)
            except Exception as e_:

                exc_type, exc_obj, exc_tb = sys.exc_info()
                print(
                    f" Exception occurred in Main :", 'error')
                print(
                    f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')

    def return_main(self):

        try:
            source_img_, source_image_json, source_image_name = self.source()
            dummy_img_ = self.dummy_source(source_img_)
            points_array = self.read_json_points(source_image_json)
            for num in range(len(points_array)):
                poly_fill_object_ = self.poly_fill_object(dummy_img_, points_array[num])
                count = 0
                extracted_object = self.extract_target_image(source_img_, poly_fill_object_, False)
                extracted_object_bounding_box_1 = self.bounding_box_object(extracted_object)
                if extracted_object_bounding_box_1:

                    while True:
                        #extracted_object = self.extract_target_image(source_img_, poly_fill_object_, False)
                        extracted_object = imutils.rotate(extracted_object, random.choice(self.angle))
                        extracted_object_bounding_box = self.bounding_box_object(extracted_object)
                        if extracted_object_bounding_box:

                            _, __, width_object, height_object = extracted_object_bounding_box
                            desired_x, desired_y, yolo_coordinates = self.obtain_valid_positions(width_object,
                                                                                            height_object)
                            target_image_position_mask = self.obj_boundary(extracted_object,
                                                                      extracted_object_bounding_box, desired_y,
                                                                      desired_x)
                            target_img_ = self.target()
                            target_image_to_add = self.make_target_img_mask(target_img_, target_image_position_mask)
                            synthetic_result_img = cv2.add(target_image_to_add, target_image_position_mask)
                            #cv2.imwrite("/home/suchith/Desktop/Project_JSW/COD/Latest/" + str(self.img_name) + ".jpg",
                                        #synthetic_result_img)
                            #with open("/home/suchith/Desktop/Project_JSW/COD/Latest/" + str(self.img_name) + ".txt",
                                      #'w') as f:
                                #f.write(f"0 {yolo_coordinates[0]} {yolo_coordinates[1]} {yolo_coordinates[2]} {yolo_coordinates[3]}")
                            count = count + 1
                            return synthetic_result_img
                            self.img_name += 1
                            #return synthetic_result_img
                            print(f"IM inside nested While loop and count is {count}")
                            if count == 3:
                                break
                        else:
                            print(f"No Extracted bounding Boxes")
                            break
                print("IM inside for loop of points")
            self.i = self.i + 1
            print(self.i)
        except Exception as e_:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(
                f" Exception occurred in Main :", 'error')
            print(
                f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')

if __name__ == "__main__":
    start = SuperImpose()
    i=0
    """
    while i < 2000:
        try:
            img = start.main()
            cv2.imwrite("/home/suchith/Desktop/Project_JSW/COD/Latest/" + str(random.randint(100000,10000000)) + ".jpg",
                        img)
        except:
            pass
    """
    try:
        start.main()
    except Exception as e_:

        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(
            f" Exception occurred in Main :", 'error')
        print(
            f"{str(e_)}\t{str(exc_type)}\t{str(exc_obj)}\t{str(exc_tb.tb_lineno)}\n", 'error')