import cv2
import numpy as np
cam=cv2.VideoCapture(0)
cam.set(3,1080)
cam.set(4,720)

def contour_process(img):
    imgg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgc=cv2.Canny(imgg,50,150)
    contours,f=cv2.findContours(imgc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    area=0
    currrect=[]
    for cont in contours:
        carea=cv2.contourArea(cont)
        peri=cv2.arcLength(cont,True)
        cnt=cv2.approxPolyDP(cont,0.02*peri,True)
        if carea<area or len(cnt)!=4:
            continue
        area=carea
        currrect=cnt
    if not len(currrect):
        return []
    return currrect

def cont_coords(cont):
    coords=[]
    for i in cont:
        coords.append(i[0])
    return np.array(coords)

def reorder(coords):
    pts=coords.reshape(4,2)
    ptsnew=np.zeros((4,1,2))
    add=pts.sum(1)
    ptsnew[0]=pts[np.argmin(add)]
    ptsnew[2]=pts[np.argmax(add)]
    diff=np.diff(pts,axis=1)
    ptsnew[1]=pts[np.argmin(diff)]
    ptsnew[3]=pts[np.argmax(diff)]
    return ptsnew
    

    

def warpimg(img,coords):
    if len(coords)!=4:
        return img
    wimg=img.copy()
    mask1=np.float32(reorder(coords))
    mask=np.float32([[0,0],[480,0],[480,720],[0,720]])
    perst=cv2.getPerspectiveTransform(mask1,mask)
    pwimg=cv2.warpPerspective(wimg,perst,(480,720))
    return pwimg


while 1:
    h,img=cam.read()
    cont=contour_process(img)
    coords=cont_coords(cont)
    wimg=warpimg(img.copy(),coords)
    for i in coords:
        cv2.circle(img,i,5,(0,125,200),5)
        cv2.putText(img,str(i),i,2,0.5,(1,0,231),2)
        
    cv2.imshow("sdcsc",img)
    cv2.imshow("sdcsck",wimg)
    if cv2.waitKey(2)==ord('q'):
        break