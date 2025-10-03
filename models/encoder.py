import torch
from typing import Dict, Optional
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel
from PIL import Image
from utils.device_manager import device_manager, DeviceMixin


class BaseEncoder(torch.nn.Module, DeviceMixin):
    """编码器基类，提供公共功能"""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        DeviceMixin.__init__(self)

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)

        if device:
            device_manager.move_model_to_device(self, torch.device(device))

    def save(self, filename: str, model_attr: str):
        """保存编码器"""
        print(f'Saving {self.__class__.__name__} to {filename}')
        torch.save({
            'model_state_dict': getattr(self, model_attr).state_dict(),
            'model_name': self.model_name,
            'cache_dir': self.cache_dir
        }, filename) 

    @classmethod
    def load(cls, filename: str):
        """加载编码器"""
        print(f'Loading {cls.__name__} from {filename}')
        checkpoint = torch.load(filename, map_location='cpu')
        return cls(
            model_name=checkpoint['model_name'],
            cache_dir=checkpoint['cache_dir']
        )


class ImageEncoder(BaseEncoder):
    """图像编码器，基于CLIP视觉模型"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_name, cache_dir, device)

        print(f'Loading {model_name} CLIP model.')
        self.clip_model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_dim = self.clip_model.config.projection_dim  # 一般为512
        # 只保留视觉模型部分
        del self.clip_model.text_model
        del self.clip_model.text_embed_dim
        del self.clip_model.text_projection

    def forward(self, images):
        device = self._get_device()
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            pixel_values = device_manager.move_tensors_to_device(inputs['pixel_values'], device=device)
        elif isinstance(images, torch.Tensor):
            pixel_values = device_manager.move_tensors_to_device(images, device=device)
        else:
            raise ValueError("Images must be either a list of PIL Images or a torch.Tensor")

        features = self.clip_model.get_image_features(pixel_values=pixel_values)
        return features

    def save(self, filename: str):
        super().save(filename, 'vision_model')

    @classmethod
    def load(cls, filename: str):
        encoder = super().load(filename)
        checkpoint = torch.load(filename, map_location='cpu')
        encoder.vision_model.load_state_dict(checkpoint['model_state_dict'])
        return encoder


class TextEncoder(BaseEncoder):
    """文本编码器，基于CLIP文本模型"""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 cache_dir: Optional[str] = None, device: Optional[str] = None):
        super().__init__(model_name, cache_dir, device)

        print(f'Loading {model_name} text model.')
        self.text_model = CLIPTextModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.feature_dim = self.text_model.config.hidden_size

    def forward(self, inputs: Dict[str, torch.Tensor], **kwargs):
        """只支持分词后的字典输入"""
        device = self._get_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_outputs = self.text_model(**inputs)
        return text_outputs

    def save(self, filename: str):
        super().save(filename, 'text_model')

    @classmethod
    def load(cls, filename: str):
        encoder = super().load(filename)
        checkpoint = torch.load(filename, map_location='cpu')
        encoder.text_model.load_state_dict(checkpoint['model_state_dict'])
        return encoder
