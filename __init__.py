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
                "clip_vision_model": ("CLIP_VISION",), # 关键：接收已加载的 CLIP Vision 模型
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "status_message")
    FUNCTION = "calculate_similarity"
    CATEGORY = "Analysis/Image"
    DESCRIPTION = "Calculates semantic similarity between two images using CLIP Vision embeddings."

    def calculate_similarity(self, image_a, image_b, clip_vision_model):
        """
        image_a/b: ComfyUI IMAGE tensor [B, H, W, C] (0-1 range)
        clip_vision_model: 预加载的 CLIP Vision 模型对象
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 确保输入在正确的设备上且格式正确
        # ComfyUI 图像通常是 [B, H, W, C], CLIP 需要 [B, C, H, W] 且归一化方式特定
        # comfy.clip_vision.encode_image 内部会处理预处理，但我们需要确保输入是 RGB
        
        # 如果输入是批次，取第一张
        if image_a.shape[0] > 1:
            image_a = image_a[0:1]
        if image_b.shape[0] > 1:
            image_b = image_b[0:1]

        # 2. 使用 ComfyUI 内置编码器提取特征
        # encode_image 返回一个对象，包含 'last_hidden_state', 'pooler_output' 等
        # 我们使用 'pooler_output' (投影后的全局向量) 进行相似度计算最准确
        try:
            embed_a = clip_vision_model.encode_image(image_a.to(device))
            embed_b = clip_vision_model.encode_image(image_b.to(device))
            
            # 提取 pooler_output (形状通常为 [1, 1024] 或 [1, 768])
            vec_a = embed_a['pooler_output']
            vec_b = embed_b['pooler_output']
            
        except Exception as e:
            return (0.0, f"Error encoding images: {str(e)}")

        # 3. 计算余弦相似度
        with torch.no_grad():
            # cosine_similarity 期望输入是 [batch, dim]
            sim_tensor = cosine_similarity(vec_a, vec_b)
            similarity_score = sim_tensor.item()

        # 4. 生成描述信息
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

# 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageCLIPSimilarity": ImageCLIPSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCLIPSimilarity": "Image CLIP Similarity"
}
