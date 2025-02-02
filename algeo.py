import cv2
import numpy as np
import scipy
from scipy.misc import imread
import _pickle as pickle
import random
import os
import matplotlib.pyplot as plt
import imageio
import math

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imageio.imread(image_path, pilmode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print ('Error:', e)
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)
    
    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)



class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path,'rb') as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.items():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        #print(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector, Euclidean):
        # getting cosine distance between search image and images database
        #print(vector)
        #v = vector.reshape(1, -1)
        v = vector.tolist()
        #print(v)
        #m = scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)
        #print(m)
        a =[]
        if (Euclidean):
            for i in range (len(self.matrix)):
                res = dstnc(v, self.matrix[i].tolist())
                a.append(res)
            a = np.array(a)
            return a
        else:
            for i in range (len(self.matrix)):
                res = dotproduct(v,self.matrix[i].tolist())/(lngth(v)*lngth(self.matrix[i].tolist()))
                a.append(res)
            a = np.array(a)
            return a


    def match(self, image_path, topn, Euclidean):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features, Euclidean)
        #print(img_distances)
        # getting top 5 records
        if (Euclidean):
            nearest_ids = np.argsort(img_distances)[:topn].tolist()
        else:
            nearest_ids = np.argsort(img_distances)[::-1][:topn].tolist()
        
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()


def show_img(path):
    img = imageio.imread(path, pilmode="RGB")
    plt.imshow(img)
    plt.show()


def dstnc(v1, v2):
    sum = 0
    for i in range (len(v1)):
        sum += (v1[i]-v2[i])**2
    return math.sqrt(sum)

def lngth(v1):
    sum = 0
    for i in range (len(v1)):
        sum += v1[i]**2
    return math.sqrt(sum)

def dotproduct(v1, v2):
    sum =0
    for i in range (len(v1)):
        sum+= v1[i] * v2[i]
    return sum

def run():
    images_path = 'pins-face-recognition\PINS\pins_Aaron Paul/'
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    # getting 3 random images
    images_path1 = 'nyoba/' 
    #files1 = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    #sample = random.sample(files,1)[0]
    a = input('Masukkan nama image: ')
    sample = 'nyoba\\' + a +'/'
    #print(sample)

    print('1. Euclidean\n2.Cosinus')
    Method = int(input('Pilih metode: '))
    batch_extractor(images_path)
    ma = Matcher('features.pck')
    Euclidean = 1==Method
    #for s in sample:
    print ('Query image ==========================================')
    show_img(sample)
    names, match = ma.match(sample, 3, Euclidean)
    print ('Result images ========================================')
    for i in range(3):
        # we got cosine distance, less cosine distance between vectors
        # more they similar, thus we subtruct it from 1 to get match value
        if (not Euclidean):
            print ('Match %s' % (match[i]))

        show_img(os.path.join(images_path, names[i]))




run()
