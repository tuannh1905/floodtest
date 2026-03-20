import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

# =========================================================================
# TÍCH HỢP GRAD-CAM CHO SEMANTIC SEGMENTATION
# =========================================================================
class SemanticGradCAM:
    """
    Tùy biến Grad-CAM cho bài toán Phân vùng.
    Mục tiêu: Tìm hiểu xem mô hình dựa vào 'điểm ảnh' nào ở feature map 
    để quyết định vùng đó là 'nước'.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Đăng ký Hooks để bắt Activation và Gradient trong quá trình chạy
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        
        # 1. Forward pass
        output = self.model(input_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
            
        pred_probs = torch.sigmoid(output)
        pred_mask = (pred_probs > 0.5).float()
        
        # 2. Tạo hàm mục tiêu (Chỉ lấy gradient từ những nơi model đoán là 'nước')
        score = (output * pred_mask).sum()
        
        # Nếu model không đoán ra giọt nước nào, bỏ qua backward
        if score.item() == 0:
            return np.zeros((input_tensor.size(2), input_tensor.size(3))), pred_probs.detach().cpu().numpy()[0, 0]
            
        # 3. Backward pass để lấy gradients
        score.backward()
        
        # 4. Trích xuất Gradients và Activations
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        # 5. Global Average Pooling trên Gradients để tìm trọng số Kênh
        weights = np.mean(gradients, axis=(1, 2))
        
        # 6. Nhân chập trọng số với Activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # 7. Xử lý ảnh CAM (ReLU -> Resize -> Normalize)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        return cam, pred_probs.detach().cpu().numpy()[0, 0]

# =========================================================================
# HÀM PHỤ TRỢ: TÍNH IOU CHO TỪNG ẢNH
# =========================================================================
def calculate_single_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).astype(np.uint8).flatten()
    target = target.astype(np.uint8).flatten()
    
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)

# =========================================================================
# HÀM PHỤ TRỢ: VẼ BẢN ĐỒ LỖI (ERROR MAP)
# =========================================================================
def create_error_map(pred, gt):
    """
    Xanh lá: Đúng (True Positive)
    Đỏ: Đoán sai thành nước (False Positive - Lỗi Semantic)
    Xanh dương: Bỏ sót nước (False Negative - Lỗi Spatial/Holes)
    """
    error_map = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    
    error_map[tp] = [0, 255, 0]     # Xanh lá
    error_map[fp] = [255, 0, 0]     # Đỏ
    error_map[fn] = [0, 0, 255]     # Xanh dương
    return error_map

# =========================================================================
# CHƯƠNG TRÌNH CHÍNH
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description='Analyze Worst Predictions and Grad-CAM')
    parser.add_argument('--dataset', type=str, default='floodkaggle', choices=['floodvn', 'floodkaggle'])
    parser.add_argument('--model', type=str, required=True, help='Tên model (vd: dabnet)')
    parser.add_argument('--weights', type=str, required=True, help='Đường dẫn tới file .pth')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--top_k', type=int, default=15, help='Số ảnh tệ nhất cần phân tích')
    parser.add_argument('--output_dir', type=str, default='analysis_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data (Dùng Validation Set, Batch size = 1 để lấy từng ảnh)
    from utils.dataloader import get_dataloaders
    _, val_loader, _ = get_dataloaders(
        dataset=args.dataset, batch_size=1, size=args.size, 
        seed=42, num_classes=1, dataset_type=args.dataset
    )
    
    # 2. Load Model
    from models import get_model
    model = get_model(args.model, num_classes=1)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Xác định Target Layer cho Grad-CAM (Khối Sâu nhất trước khi kết hợp)
    # Tùy kiến trúc, có thể thay đổi. Với ProDABNet V5, layer tốt nhất là DAB_Block_2
    try:
        target_layer = model.DAB_Block_2[-1] 
    except AttributeError:
        # Fallback nếu dùng model khác
        target_layer = list(model.children())[-2] 

    grad_cam = SemanticGradCAM(model, target_layer)

    print(f"\n🚀 Đang phân tích {len(val_loader)} ảnh từ tập Validation...")
    results = []

    # 3. Chạy qua toàn bộ tập Val để chấm điểm
    for idx, (img_tensor, mask_tensor) in enumerate(tqdm(val_loader)):
        img = img_tensor.to(device)
        mask_np = mask_tensor.numpy()[0, 0] if mask_tensor.dim() == 4 else mask_tensor.numpy()[0]
        
        with torch.no_grad():
            output = model(img)
            if isinstance(output, (list, tuple)):
                output = output[0]
            pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]
            
        iou = calculate_single_iou(pred_prob, mask_np)
        
        # Lưu lại thông tin (Tensor được chuyển về CPU để giải phóng VRAM)
        results.append({
            'id': idx,
            'img_tensor': img_tensor.cpu(),
            'gt': mask_np,
            'iou': iou
        })

    # 4. Lọc top K ảnh có mIoU thấp nhất
    results.sort(key=lambda x: x['iou'])
    worst_cases = results[:args.top_k]

    print(f"\n🔬 Đang vẽ Heatmap cho {args.top_k} ảnh tệ nhất...")
    
    for rank, item in enumerate(worst_cases):
        img_tensor = item['img_tensor'].to(device)
        gt = item['gt']
        iou = item['iou']
        
        # Sinh Grad-CAM
        cam, pred_prob = grad_cam.generate_cam(img_tensor)
        pred_mask = (pred_prob > 0.5).astype(np.uint8)
        
        # Đưa ảnh Gốc về định dạng hiển thị (H, W, 3)
        img_show = img_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        # Nếu dataloader chia 255, ta nhân lại
        if img_show.max() <= 1.0:
            img_show = (img_show * 255).astype(np.uint8)
            
        # Overlay CAM lên ảnh gốc
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_show, 0.5, cam_heatmap, 0.5, 0)
        
        # Lập Bản đồ lỗi
        error_map = create_error_map(pred_mask, gt)

        # 5. Vẽ và Lưu hình
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Rank: #{rank+1} Tệ nhất | Ảnh ID: {item["id"]} | mIoU: {iou:.4f}', fontsize=16)
        
        axes[0].imshow(img_show)
        axes[0].set_title("Ảnh Gốc")
        
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title("Nhãn (Ground Truth)")
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Dự đoán (Prediction)")
        
        axes[3].imshow(error_map)
        axes[3].set_title("Bản Đồ Lỗi (Lá: Đúng | Đỏ: Nhầm | Dương: Sót)")
        
        axes[4].imshow(overlay)
        axes[4].set_title("Grad-CAM (Sự chú ý của Model)")
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f'worst_{rank+1}_iou_{iou:.3f}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"\n✅ Hoàn tất! Đã lưu {args.top_k} báo cáo phân tích tại thư mục: {args.output_dir}/")

if __name__ == '__main__':
    main()