from fastapi import FastAPI, File, UploadFile,Request
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
class DataRequest(BaseModel):
    text_data: str
    numpy_data: list

app = FastAPI()

# Đường dẫn lưu trữ tệp ảnh
UPLOAD_FOLDER = "uploads"
dir=None
save_image=None
count=0
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.post("/upload/")
async def create_upload_file(request:Request):
    global dir,save_image,count
    data = await request.body()
    # print(data)
    if len(data)>100:
        image=io.BytesIO(data)
        save_im=Image.open(image)
        image_np=np.array(save_im,dtype=np.uint8)
        save_image=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
    else:
        dir=data.decode()
        print("Full name: ",data.decode())
    path=os.path.join(UPLOAD_FOLDER,dir)
    if  not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    save_path=os.path.join(path,str(count)+".jpg")
    cv2.imwrite(save_path,save_image)
    count+=1
    
    # # Lưu tệp vào ổ đĩa hoặc xử lý nó theo ý muốn
    # with open(file.filename, "wb") as f:
    #     f.write(await file.read())

 
