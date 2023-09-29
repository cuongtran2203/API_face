#decode: utf8
from fastapi import FastAPI, File, UploadFile,Request
import uvicorn
from fastapi.responses import JSONResponse,HTMLResponse
import shutil
import os
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
import cv2
import base64
from loguru import logger
from src.get_emb import Face_recognition
class DataRequest(BaseModel):
    text_data: str
    numpy_data: list
app = FastAPI()
model=Face_recognition()
# Đường dẫn lưu trữ tệp ảnh
UPLOAD_FOLDER = "uploads"
UPLOAD_EMB_FOLDER="embs"
UPLOAD_FACE_FOLDER="faces"
dir=None
save_image=None
count=0
fullname=None
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_EMB_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FACE_FOLDER, exist_ok=True)

@app.post("/upload/")
async def create_upload_file(request:Request):
    global dir,save_image,count,fullname
    data = await request.body()
    # print(data)
    if len(data)>100:
        image=io.BytesIO(data)
        save_im=Image.open(image)
        image_np=np.array(save_im,dtype=np.uint8)
        save_image=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    else:
        dir=str(data.decode())
        fullname=data.decode()
        print("Full name: ",data.decode())
    path=os.path.join(UPLOAD_FOLDER,str(data.decode()))
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    save_path=os.path.join(path,str(count)+".jpg")
    cv2.imwrite(save_path,save_image)
    count+=1  
    if data.decode()=="Getgo":
        for root,dirs,files in os.walk(UPLOAD_FOLDER):
            if len(dirs)<1:
                print(root)
                emb_list=[]
                for file in files:
                    img=cv2.imread(os.path.join(root,file))
                    face_list,emb=model.run(img)
                    for face in face_list:
                        if not os.path.exists(os.path.join(UPLOAD_FACE_FOLDER,root.split("/")[1])):
                            os.makedirs(os.path.join(UPLOAD_FACE_FOLDER,root.split("/")[1]),exist_ok=True)
                        else:
                            return {"messsage": "folder existed"}
                        cv2.imwrite(os.path.join(UPLOAD_FACE_FOLDER,root.split("/")[1],root.split("/")[2]),np.array(face,dtype=np.uint8))
                    emb_list.append(emb)
                emb_list=np.asarray(emb_list)
                if not os.path.exists(os.path.join(UPLOAD_EMB_FOLDER,root.split("/")[1]+".npy")):
                    with open(os.path.join(UPLOAD_EMB_FOLDER,root.split("/")[1]+".npy"),"wb") as f :
                        np.save(f,emb_list)
                        logger.info("Saved emb successfully")
                else:
                    return {"messsage": "folder existed"}

                

    # # Lưu tệp vào ổ đĩa hoặc xử lý nó theo ý muốn
    # with open(file.filename, "wb") as f:
    #     f.write(await file.read())
if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8000)
 
