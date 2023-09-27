from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QDir, Qt, QUrl, QSize
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, 
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QStatusBar,QLineEdit,QTextEdit,QMessageBox,QFrame)
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets, uic
from app_QT import Ui_MainWindow
import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import requests
import io
import json
import numpy as np
from PIL import Image
RTSP_LINK=""

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self,RTSP_LINK):
        super().__init__()
        self._run_flag = True
        self.RTSP_Link=RTSP_LINK

    def run(self):
        # capture from web cam
        cap = cv2.VideoCapture(self.RTSP_Link)
        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False

        #self.wait()
class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('app.ui', self) # Load the .ui file
        self.windows=Ui_MainWindow()
        self.button_Capture=self.findChild(QPushButton,"Capture")
        self.button_Exit=self.findChild(QPushButton,"Exit")
        self.label_RTSP=self.findChild(QLineEdit,"RTSP")
        self.label_name=self.findChild(QTextEdit,"FullName")
        self.page=self.findChild(QWidget,"page")
        self.page_2=self.findChild(QWidget,"page_2")
        self.frame=self.findChild(QLabel,"frame")
        self.current_page_index = 1
        self.pages=[self.page,self.page_2]
        # self.setCentralWidget(self.pages[self.current_page_index])
        self.button_Next=self.findChild(QPushButton,"Next")
        self.button_Back=self.findChild(QPushButton,"Back")
        self.cap=None        
        self.button_Capture.clicked.connect(self.Capture_Image)
        self.button_Next.clicked.connect(self.next_tab)
        self.button_Back.clicked.connect(self.back_tab)
        self.button_Exit.clicked.connect(self.exit)
        self.IMG=None
        self.RTSP=""
        self.FullName=""
        base_url = 'http://127.0.0.1:8000'
        endpoint='upload/'
        api_url=base_url+"/"+endpoint
        self.api_server=api_url
        

    def closeEvent(self):
        self.thread.stop()
        self.__init__()
        self.button_Next.clicked.connect(self.activate_thread)
        #event.accept()

    def activate_thread(self,RTSP):
        # create the video capture thread
        self.thread = VideoThread(RTSP)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

        self.button_Exit.clicked.connect(self.thread.stop)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        self.IMG=cv_img
        qt_img = self.convert_cv_qt(cv_img)
        self.frame.setPixmap(qt_img)
    def Capture_Image(self):
        Fullname=self.label_name.toPlainText()
        self.FullName=Fullname
        print(Fullname)
        if self.IMG is not None :
            self.send_image(self.IMG)
            QMessageBox.information(self,"Capture Image","Capture Image Sucessfully")
        
    def next_tab(self):
        rtsp=self.label_RTSP.text()
        self.RTSP=rtsp
       
        print(self.RTSP)
        if len(rtsp)<5:
            QMessageBox.warning(self,"Warning","Please insert RTSP or TCP/IP link")
        else :
            self.activate_thread(self.RTSP)
            self.switch_tab()
    def exit(self):
        sys.exit()
    def back_tab(self):
        Fullname=self.label_name.toPlainText()
        self.FullName=Fullname
        print(Fullname)
        if len(Fullname)<1:
            QMessageBox.warning(self,"Warning","Please insert name")
        else:
            self.switch_backtab()
    def switch_tab(self):
        # Ẩn trang hiện tại
        self.pages[self.current_page_index].hide()

        # Chuyển đổi index để hiển thị trang khác
        self.current_page_index = (self.current_page_index + 1) % len(self.pages)

        # Hiển thị trang mới
        self.setCentralWidget(self.pages[self.current_page_index])
        self.pages[self.current_page_index].show()
    def switch_backtab(self):
        # Ẩn trang hiện tại
        self.pages[self.current_page_index].hide()

        # Chuyển đổi index để hiển thị trang khác
        self.current_page_index = (self.current_page_index -1) % len(self.pages)

        # Hiển thị trang mới
        self.setCentralWidget(self.pages[self.current_page_index])
        self.pages[self.current_page_index].show()
    def send_image(self,img:np.ndarray):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        send_image=Image.fromarray(img)
        image2bytes=io.BytesIO()
        send_image.save(image2bytes,format="JPEG")
        image2bytes.seek(0)

        response = requests.post(self.api_server, data=image2bytes)
        response = requests.post(self.api_server, data=self.FullName)
        if response.status_code == 200:
            QMessageBox.information(self,"Infor","Uploaded Image to Severs")
        else:
            QMessageBox.warning(self,"Warning",str(response.status_code))
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(600, 500, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
        

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())