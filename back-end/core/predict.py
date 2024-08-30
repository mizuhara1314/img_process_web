import cv2
import numpy as np  
def gauss(pic):
    o=cv2.imread(pic) 
    r=cv2.GaussianBlur(o, (5,5),0,0) 
    return r


def enhance(pic):        
    
    image = cv2.imread(pic, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b, g, r])
    return image

def edge(pic):
        img = cv2.imread(pic, 0)
        
        x = cv2.Sobel(img,cv2.CV_16S,1,0)
        y = cv2.Sobel(img,cv2.CV_16S,0,1)
        
        absX = cv2.convertScaleAbs(x)   # 转回uint8
        absY = cv2.convertScaleAbs(y)
        
        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)   
        return dst

def predict(dataset, mode, ext):
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    print(x)
    print(file_name)
    img_y = gauss(x)
    if mode == 2:
       img_y = enhance(x)
    elif mode == 3:
       img_y = edge(x)
    
    cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
    