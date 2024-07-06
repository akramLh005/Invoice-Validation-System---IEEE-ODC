import torch
import torchvision
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from functools import partial

# Device configuration for utilizing GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DetectionModel:
    """
    A class to define the backbone of the detection system using a modified RetinaNet model.

    Attributes:
        model (torch.nn.Module): The object detection model.
    
    Methods:
        __init__(self, model_path, num_classes=3): Initializes the model with pre-trained weights and a specified number of classes.
        create_model(self, num_classes): Constructs the model with a custom classification head.
        predict(self, image_tensor): Predicts the objects in the given image.
    """
    def __init__(self, model_path, num_classes):
        """
        Initializes the detection model with pre-trained weights and adjusts the classification head for the given number of classes.

        Args:
            model_path (str): Path to the model's state dictionary.
            num_classes (int): Number of classes to detect.
        """
        self.model = self.create_model(num_classes)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE).eval()

    def create_model(self, num_classes):
        """
        Constructs the modified RetinaNet model using a pre-trained backbone and a customized classification head.

        Args:
            num_classes (int): The number of classes to classify.

        Returns:
            torch.nn.Module: The constructed object detection model.
        """
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        return model

    def predict(self, image_tensor):
        """
        Perform object detection on the input image tensor.

        Args:
            image_tensor (torch.Tensor): The preprocessed image tensor.

        Returns:
            dict: The detection outputs including bounding boxes, labels, and scores.
        """
        with torch.no_grad():
            outputs = self.model(image_tensor)
        return outputs
