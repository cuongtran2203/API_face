from fastapi import FastAPI, File, UploadFile,Request
from fastapi.responses import JSONResponse,HTMLResponse
import shutil
import os
from pydantic import BaseModel
import base64
import io
from PIL import Image
import aiofiles

app = FastAPI()
class ImageData(BaseModel):
    image_data: str  # Dùng str cho base64-encoded image data
# Đường dẫn lưu trữ tệp ảnh
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def create_upload_file(request:Request):
    form_data = await request.form()
    file = form_data["image"]  # "image" là tên của trường trong form

    # Lưu tệp vào ổ đĩa hoặc xử lý nó theo ý muốn
    with open(file.filename, "wb") as f:
        f.write(await file.read())

 