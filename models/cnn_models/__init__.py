"""CNN-based visual feature extractors for medical imaging."""

from models.cnn_models.densenet import MedicalDenseNet
from models.cnn_models.resnet import MedicalResNet

__all__ = ["MedicalDenseNet", "MedicalResNet"]
