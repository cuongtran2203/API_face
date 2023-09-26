from core.face_detect import *
import torch
import cv2
import numpy as np
from core.config import cfg_mnet
from core.prior_box  import PriorBox
from core.ultils import decode,decode_landm,py_cpu_nms
import onnxruntime as ort
class Face_Detection():
    def __init__(self,model_path=None) -> None:
        ##call model
        self.cfg=cfg_mnet
        self.net=ort.InferenceSession(model_path)
    def preprocess(self,img_raw):
        img = np.float32(img_raw)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        scale = scale.to("cpu")
        return img,scale
    def postprocess(self,output,img,scale):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loc, conf, landms=output
        # print(loc.shape)
        # print(conf.shape)
        # print(landms.shape)
        conf=torch.as_tensor(conf)
        loc=torch.as_tensor(loc)
        landms=torch.as_tensor(landms)
        priorbox = PriorBox(self.cfg, image_size=(640,640))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        # prior_data=prior_data.detach().numpy()
        # print(priors.shape)
        resize=1
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.4)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        #print(order)
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:5000, :]
        landms = landms[:5000, :]
        dets = np.concatenate((dets, landms), axis=1)
        return dets
    def detect(self,img):
        input,scale=preprocess(img)
        print(input.shape)
        onnx_input = {self.net.get_inputs()[0].name:input.detach().numpy()}
        output= self.net.run(None,onnx_input)
        dets=self.postprocess(output,input,scale)
        faces_list=[]
        if dets is not None :
            for b in dets:
                if b[4] < 0.6:
                    continue
                b = list(map(int, b))
                face=img[b[1]:b[3],b[0]:b[2]]
                faces_list.append(face)
            return faces_list
        else:
            return []
        
            
        

if __name__=="__main__":
    model=Face_Detection(model_path="src/weights/Face_Detector.onnx")
    img=cv2.imread("a7.jpg")
    img=cv2.resize(img,(640,640))
    tic=time.time()
    faces=model.detect(img)
    print("Time processing: {}".format(time.time()-tic))
    cv2.imwrite("face.jpg",faces[0])
    print("faces")