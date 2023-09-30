import torch 
import torchvision
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import timm
EMOTION_DICT = {0: "neutral", 1: "happiness", 2: "surprise", 3: "sadness", 4: "anger", 5: "disgust", 6: "fear"}
class Face_expression_recognition():
    def __init__(self,model_path,model_arch="mobilenetv2_140", pretrained=False, num_classes: int = 7):
        self.device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_arch, pretrained=pretrained, num_classes=num_classes)
        self.load_weights(model_path)
        self.model.to(self.device).eval()
        self.size=224
    def load_weights(self, path_to_weights: str) -> None:
        cp = torch.load(path_to_weights)
        state_dict = cp["state_dict"]
        state_dict = {k.replace("model.model.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
    def preprocess(self,img):
        fer_transforms = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet values
            ]
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image = Image.fromarray(gray)
        image = fer_transforms(image).float()
        image_tensor = image.unsqueeze_(0)
        input = image_tensor.to(self.device)
        return input
    def infer(self):
        if self.model is not None:
            with torch.no_grad():
                output = self.model(input)
        probs = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy()
        return {EMOTION_DICT[probs[0].argmax()]: np.amax(probs[0])}


        
