import torch
import numpy as np
from torch.nn.functional import cosine_similarity
import comfy.clip_vision
import folder_paths
from PIL import Image

class ImageCLIPSimilarity:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),          # 接收第一张图片
                "image_b": ("IMAGE",),          # 接收第二张图片
                # 关键：用下拉框选择CLIP Vision模型名称，而非直接传模型对象
                "clip_vision_name": (folder_paths.get_filename_list("clip_vision"),),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "status_message")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Analysis/Image"
    DESCRIPTION = "Calculates semantic similarity between two images using CLIP Vision embeddings."

    def calculate_similarity(self, image_a, image_b, clip_vision_name):
        """
        核心修复：
        1. 手动加载CLIP Vision模型，避免参数不匹配
        2. 标准化图像预处理流程
        3. 兼容不同版本ComfyUI的embedding键名
        """
        try:
            # 1. 加载CLIP Vision模型（从下拉框选择的名称加载）
            clip_vision_path = folder_paths.get_full_path("clip_vision", clip_vision_name)
            clip_vision = comfy.clip_vision.load_clip_vision(clip_vision_path)
            device = clip_vision.device  # 模型所在设备（自动适配GPU/CPU）

            # 2. 图像预处理：ComfyUI IMAGE → PIL → CLIP输入格式
            def preprocess_comfy_image(img_tensor):
                # 处理批次：取第一张图片 [B, H, W, C] → [H, W, C]
                if img_tensor.shape[0] > 0:
                    img_tensor = img_tensor[0]
                # 0-1 转 0-255 并转为numpy数组
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                # 转为PIL图像并强制RGB（去除Alpha通道）
                pil_img = Image.fromarray(img_np).convert("RGB")
                # 使用CLIP官方预处理
                processed_img = clip_vision.preprocess(pil_img).unsqueeze(0).to(device)
                return processed_img

            # 3. 处理两张输入图片
            img_a = preprocess_comfy_image(image_a)
            img_b = preprocess_comfy_image(image_b)

            # 4. 提取CLIP特征向量
            with torch.no_grad():
                embed_a = clip_vision.encode_image(img_a)
                embed_b = clip_vision.encode_image(img_b)
                
                # 兼容不同版本的embedding键名
                vec_a = embed_a.get("image_embeds", embed_a.get("pooler_output"))
                vec_b = embed_b.get("image_embeds", embed_b.get("pooler_output"))

                # 压缩维度：确保是 [dim] 形状
                vec_a = vec_a.squeeze()
                vec_b = vec_b.squeeze()

            # 5. 计算余弦相似度
            if vec_a.dim() == 1 and vec_b.dim() == 1:
                # 单向量计算
                similarity = cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0), dim=1).item()
            else:
                # 兼容批次维度
                similarity = cosine_similarity(vec_a, vec_b, dim=-1).mean().item()

            # 6. 生成状态信息
            status = f"相似度得分: {similarity:.4f}"
            if similarity > 0.90:
                status += " (几乎相同)"
            elif similarity > 0.80:
                status += " (高度相似)"
            elif similarity > 0.65:
                status += " (中度相似)"
            elif similarity > 0.45:
                status += " (低度相似)"
            else:
                status += " (不相关)"

            return (similarity, status)

        except Exception as e:
            error_info = f"错误详情: {str(e)}"
            return (0.0, error_info)

# 注册节点（必须确保名称一致）
NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarity": ImageCLIPSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarity": "Image CLIP Similarity"
}
