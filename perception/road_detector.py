import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet101

class RoadDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = deeplabv3_resnet101(pretrained=True).eval().to(self.device)
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def is_on_road(self, image: Image.Image) -> bool:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)['out']
        pred = output.argmax(1).squeeze().cpu().numpy()

        h, w = pred.shape
        center_region = pred[h//3:2*h//3, w//3:2*w//3]
        road_ratio = (center_region == 0).sum() / center_region.size
        return road_ratio > 0.3