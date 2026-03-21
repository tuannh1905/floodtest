import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss cho bài toán Phân vùng nhị phân (Binary Segmentation).
    Tập trung đạo hàm vào các "pixel khó" (mép nước, bóng râm) và giảm 
    trọng số của các "pixel dễ" (vùng nước/nền quá rõ ràng).
    """
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Tính BCE Loss cơ bản cho từng pixel (chưa tính trung bình)
        # Hàm này nhận raw logits (chưa qua sigmoid) nên rất an toàn
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Lấy xác suất dự đoán p_t thông qua BCE (toán học: p_t = exp(-BCE))
        pt = torch.exp(-bce_loss)
        
        # Tính hệ số điều chỉnh Focal: (1 - p_t)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        # Áp dụng trọng số Alpha để cân bằng lớp (Lũ / Nền)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

def build_loss(num_classes=1):
    # Khởi tạo Focal Loss với các tham số chuẩn mực nhất
    return FocalLoss(alpha=0.25, gamma=2.0, num_classes=num_classes)