# Library for processing the images in a jiffy
# Created by Joel John Mathew a.k.a 'FORTFANOP' (www.github.com/FORTFANOP)
# Please Star this repo if you find this helpful


try:
    import cv2
    import os
    import pickle
    import numpy as np
    print("Dependencies loaded successfully...")
except:
    print("Required libraries not found")


class PreprocessImage(object):

    def __init__(self,PATH='', IMAGE_SIZE = 50, X_NAME='X_Data', Y_NAME='Y_Data'):
        self.PATH = PATH
        self.IMAGE_SIZE = IMAGE_SIZE
        self.X_NAME = X_NAME
        self.Y_NAME = Y_NAME

        self.image_data = []
        self.x_data = []
        self.y_data = []
        self.CATEGORIES = []

        # List of all categories in the folder
        self.list_categories = []

    def get_categories(self):
        for path in os.listdir(self.PATH):
            if '.DS_Store' in path:
                pass
            else:
                self.list_categories.append(path)
        print("Found ", len(self.list_categories), " Categories: ",self.list_categories,'\n')
        return self.list_categories

    def Process_Image(self):
        try:
            """
            Return Numpy array of image
            :return: X_Data, Y_Data
            """
            self.CATEGORIES = self.get_categories()
            for categories in self.CATEGORIES:                                                  # Iterate over categories

                train_folder_path = os.path.join(self.PATH, categories)                         # Folder Path
                class_index = self.CATEGORIES.index(categories)                                 # this will get index for classification

                for img in os.listdir(train_folder_path):                                       # This will iterate in the Folder
                    new_path = os.path.join(train_folder_path, img)                             # image Path

                    try:                    # if any image is corrupted
                        image_data_temp = cv2.imread(new_path,cv2.IMREAD_GRAYSCALE)                 # Read Image as numbers
                        # image_data_temp = cv2.Canny(image_data_temp, 80, 80)                      # image canny (edge detection)
                        image_temp_resize = cv2.resize(image_data_temp,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                        self.image_data.append([image_temp_resize,class_index])
                    except:
                        pass

            data = np.asanyarray(self.image_data)

            # Iterate over the Data
            for x in data:
                self.x_data.append(x[0])        # Get the X_Data
                self.y_data.append(x[1])        # get the label

            X_Data = np.asarray(self.x_data) / (255.0)      # Normalize Data
            Y_Data = np.asarray(self.y_data)

            # reshape x_Data

            X_Data = X_Data.reshape(-1, self.IMAGE_SIZE, self.IMAGE_SIZE, 1)

            return X_Data, Y_Data
        except:
            print("Failed to run Process Image")

    def pickle_image(self):
        # Call the Function and Get the Data
        X_Data,Y_Data = self.Process_Image()

        # Write the Entire Data into a Pickle File
        pickle_out = open(self.X_NAME,'wb')
        pickle.dump(X_Data, pickle_out)
        pickle_out.close()

        # Write the Y Label Data
        pickle_out = open(self.Y_NAME, 'wb')
        pickle.dump(Y_Data, pickle_out)
        pickle_out.close()

        print("Successfully preprocessed images!")
        return X_Data,Y_Data

    def load_dataset(self):
        print('Loading File and Dataset...')
        X_Data,Y_Data = self.pickle_image()
        return X_Data,Y_Data
