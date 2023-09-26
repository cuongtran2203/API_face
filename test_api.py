import requests

def upload_image_to_api(api_url, image_path):
    try:
        # Tạo một yêu cầu POST và tải lên tệp ảnh
        with open(image_path, "rb") as file:
            files = {"file": (image_path, file)}
            response = requests.post(api_url, files=files)

        # Kiểm tra phản hồi từ API
        if response.status_code == 200:
            print("Tải ảnh lên thành công!")
        else:
            print("Lỗi khi tải ảnh lên API. Mã lỗi:", response.status_code)
    except Exception as e:
        print("Lỗi khi gửi yêu cầu tới API:", str(e))

# Gọi hàm với URL của API và đường dẫn tới tệp ảnh
base_url = 'http://127.0.0.1:8000'
endpoint='upload/'
api_url=base_url+"/"+endpoint
image_path = "ram.jfif"
upload_image_to_api(api_url, image_path)