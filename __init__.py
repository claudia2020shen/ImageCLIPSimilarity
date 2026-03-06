import torch
from torch.nn.functional import cosine_similarity
import comfy.clip_vision
import folder_paths

class ImageCLIPSimilarity:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "clip_vision_model": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "status_message")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Analysis/Image"
    DESCRIPTION = "Calculates semantic similarity between two images using CLIP Vision embeddings."

    def calculate_similarity(self, image_a, image_b, clip_vision_model):
        """
        修复点：
        1. 处理ComfyUI图像张量格式（维度转换+数值归一化）
        2. 确保所有张量与模型在同一设备
        3. 增加异常捕获和维度校验
        """
        # 获取模型所在设备（关键：以模型设备为准，而非手动指定）
        device = next(clip_vision_model.parameters()).device if hasattr(clip_vision_model, 'parameters') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # --------------- 核心修复：图像预处理 ---------------
            def preprocess_image(image):
                # 1. 取批次第一张图片，确保维度 [1, H, W, C]
                if image.shape[0] > 1:
                    image = image[0:1]
                # 2. 维度转换：[B, H, W, C] → [B, C, H, W]（通道在前）
                image = image.permute(0, 3, 1, 2)
                # 3. 数值归一化：0-1 → -1到1（CLIP要求）
                image = (image * 2.0) - 1.0
                # 4. 确保是RGB（移除Alpha通道，如果有）
                if image.shape[1] == 4:
                    image = image[:, :3, :, :]
                # 5. 移到模型设备
                return image.to(device)

            # 预处理两张图片
            img_a = preprocess_image(image_a)
            img_b = preprocess_image(image_b)

            # --------------- 提取CLIP Embedding ---------------
            # 使用ComfyUI内置的编码函数，传入预处理后的图像
            embed_a = clip_vision_model.encode_image(img_a)
            embed_b = clip_vision_model.encode_image(img_b)

            # 提取pooler_output并确保在同一设备
            vec_a = embed_a['pooler_output'].to(device)
            vec_b = embed_b['pooler_output'].to(device)

            # --------------- 计算余弦相似度 ---------------
            with torch.no_grad():
                # 确保维度为 [1, dim]，避免维度不匹配
                vec_a = vec_a.squeeze(0) if vec_a.dim() > 2 else vec_a
                vec_b = vec_b.squeeze(0) if vec_b.dim() > 2 else vec_b
                # 计算余弦相似度（dim=1表示按特征维度计算）
                sim_tensor = cosine_similarity(vec_a, vec_b, dim=1)
                # 取平均值（处理批次维度）
                similarity_score = torch.mean(sim_tensor).item()

            # --------------- 生成状态信息 ---------------
            status = f"CLIP Similarity: {similarity_score:.4f}"
            if similarity_score > 0.90:
                status += " (几乎相同 / Nearly Identical)"
            elif similarity_score > 0.80:
                status += " (高度相似 / Highly Similar)"
            elif similarity_score > 0.65:
                status += " (中度相似 / Moderately Similar)"
            elif similarity_score > 0.45:
                status += " (低度相似 / Low Similarity)"
            else:
                status += " (不相关 / Unrelated)"

            return (similarity_score, status)

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return (0.0, error_msg)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarity": ImageCLIPSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarity": "Image CLIP Similarity"
}
