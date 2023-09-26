
from face_core_recog import ArcFace
from multiprocessing import Process
import sys
import datetime
import os
import numpy as np
import cv2
from Detect_retinaface import Face_Detection
class Face_recognition():
    def __init__(self):
        self.face_rec=ArcFace()
        self.face_detect=Face_Detection(model_path="src/weights/Face_Detector.onnx")
    def compare_2_emb(self,emb2):
        root="DB"
        list_dir=os.listdir(root)
        emb_vector=[]
        dist_min=0
        label=[]
        for file in list_dir:
            with open(os.path.join(root,file),"rb") as f :
                emb1=np.load(f)
            for emb in emb1:
                emb_vector.append([emb,file.split('.')[0]])
        min=9
        for emb_labels in emb_vector:
            
            dist_min=self.face_rec.get_distance_embeddings(emb_labels[0],emb2)
            
            if dist_min<min:
                min=dist_min
                label=emb_labels[1]
        if min>0.6:
            print(min)
            label="None"
        return label

    def run(self,frame):
        img_raw=cv2.resize(frame,(640,640))
        face_list= self.face_detect.detect(img_raw)
        faces=[]
        list_emb=[]
        for face in face_list:
            emb1=self.face_rec.calc_emb(face)
            list_emb.append(emb1)

                    
        return face_list,list_emb
if __name__ =="__main__":
    model=Face_recognition()
    img=cv2.imread("a7.jpg")
    face_list,list_emb=model.run(img)
    print(list_emb)
