import os
import sys
import torch
import random
import numpy as np
import argparse
import json
import warnings
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import time

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    torch.use_deterministic_algorithms(True, warn_only=True)
    warnings.filterwarnings('ignore', category=UserWarning)
    print(f"✓ Seed set to {seed} (STRICT MODE)")


class FloodNetDataset(Dataset):
    def __init__(self, root_dir, split='train', size=512, seed=42):
        self.size = size
        self.seed = seed
        self.split = split
        
        self.root_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.root_dir, f'{split}-org-img')
        self.labels_dir = os.path.join(self.root_dir, f'{split}-label-img')
        
        self.images = sorted([
            img for img in os.listdir(self.images_dir)
            if img.endswith('.jpg')
        ])
        
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        filename_hash = hash(img_name) % 1000000
        sample_seed = self.seed + filename_hash + worker_id * 10000000
        
        np.random.seed(sample_seed)
        random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        base_name = os.path.splitext(img_name)[0]
        mask_name = f'{base_name}_lab.png'
        mask_path = os.path.join(self.labels_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image'].float() / 255.0
        mask = transformed['mask'].long()
        
        return image, mask


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_seed)


def get_dataloaders(dataset_path, batch_size=4, size=256, seed=42):
    train_dataset = FloodNetDataset(dataset_path, 'train', size, seed)
    val_dataset = FloodNetDataset(dataset_path, 'val', size, seed)
    test_dataset = FloodNetDataset(dataset_path, 'test', size, seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        worker_init_fn=seed_worker,
        persistent_workers=False
    )
    
    return train_loader, val_loader, test_loader


def calculate_miou(all_preds, all_labels, num_classes=10):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    preds_class = np.argmax(all_preds, axis=1)
    
    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)
    
    ious = []
    for cls in range(num_classes):
        pred_mask = (preds_flat == cls)
        label_mask = (labels_flat == cls)
        
        intersection = np.logical_and(pred_mask, label_mask).sum()
        union = np.logical_or(pred_mask, label_mask).sum()
        
        if union > 0:
            ious.append(float(intersection / union))
    
    return float(np.mean(ious)) if ious else 0.0


def calculate_dice_score(all_preds, all_labels, num_classes=10):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    preds_class = np.argmax(all_preds, axis=1)
    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)
    
    dice_scores = []
    for cls in range(num_classes):
        pred_mask = (preds_flat == cls)
        label_mask = (labels_flat == cls)
        
        intersection = np.logical_and(pred_mask, label_mask).sum()
        dice = (2.0 * intersection) / (pred_mask.sum() + label_mask.sum() + 1e-8)
        
        if pred_mask.sum() > 0 or label_mask.sum() > 0:
            dice_scores.append(float(dice))
    
    return float(np.mean(dice_scores)) if dice_scores else 0.0


def calculate_pixel_accuracy(all_preds, all_labels):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    preds_class = np.argmax(all_preds, axis=1)
    preds_flat = preds_class.reshape(-1)
    labels_flat = all_labels.reshape(-1)
    
    correct = (preds_flat == labels_flat).sum()
    total = preds_flat.size
    
    return float(correct / total)


def calculate_model_complexity(model, input_size=(1, 3, 512, 512), device='cuda'):
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    dummy_input = torch.randn(input_size).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        memory_allocated = 0.0
    
    total_ops = 0
    hooks = []
    
    def count_ops_hook(module, input, output):
        nonlocal total_ops
        if isinstance(module, torch.nn.Conv2d):
            batch_size = input[0].size(0)
            output_height = output.size(2)
            output_width = output.size(3)
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
            output_ops = output_height * output_width * module.out_channels
            total_ops += batch_size * kernel_ops * output_ops
        elif isinstance(module, torch.nn.Linear):
            batch_size = input[0].size(0)
            total_ops += batch_size * module.in_features * module.out_features
    
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(count_ops_hook))
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    gflops = total_ops / 1e9
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'memory_mb': memory_allocated,
        'gflops': gflops
    }


def measure_inference_time(model, input_size=(1, 3, 512, 512), device='cuda', warmup=10, iterations=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    times = []
    
    with torch.no_grad():
        for _ in range(iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    fps = 1.0 / avg_time
    latency_ms = avg_time * 1000
    
    return {
        'avg_time_s': avg_time,
        'std_time_s': std_time,
        'min_time_s': min_time,
        'max_time_s': max_time,
        'fps': fps,
        'latency_ms': latency_ms
    }


def train_floodnet(model_name, loss_name, size, epochs, batch_size, lr, 
                   dataset_path, output_path, seed):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    set_seed(seed)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_path, batch_size, size, seed
    )
    
    print(f"✓ DataLoaders: num_workers=4, persistent_workers=False")
    print(f"  Train: {len(train_loader)} batches (drop_last=True)")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    
    set_seed(seed)
    
    from models import get_model
    model = get_model(model_name, num_classes=10, seed=seed)
    model = model.to(device)
    model = model.float()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model: {model_name}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print(f"\n{'='*70}")
    print("MODEL COMPLEXITY ANALYSIS")
    print(f"{'='*70}")
    
    complexity = calculate_model_complexity(model, input_size=(1, 3, size, size), device=device)
    
    print(f"Total Parameters:    {complexity['total_params']:,}")
    print(f"Trainable Parameters: {complexity['trainable_params']:,}")
    print(f"Model Size:          {complexity['model_size_mb']:.2f} MB")
    print(f"Peak Memory Usage:   {complexity['memory_mb']:.2f} MB")
    print(f"GFLOPs:              {complexity['gflops']:.4f}")
    
    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE")
    print(f"{'='*70}")
    
    inference_stats = measure_inference_time(model, input_size=(1, 3, size, size), device=device)
    
    print(f"Average Inference Time: {inference_stats['avg_time_s']*1000:.4f} ms (± {inference_stats['std_time_s']*1000:.4f} ms)")
    print(f"Min Inference Time:     {inference_stats['min_time_s']*1000:.4f} ms")
    print(f"Max Inference Time:     {inference_stats['max_time_s']*1000:.4f} ms")
    print(f"FPS:                    {inference_stats['fps']:.2f}")
    print(f"Latency:                {inference_stats['latency_ms']:.4f} ms")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    from losses import get_loss
    criterion = get_loss(loss_name, num_classes=10)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=False)
    
    print(f"✓ Optimizer: Adam (fused=False for determinism)")
    print(f"✓ Loss: {loss_name}")
    
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, f'{model_name}_{loss_name}_floodnet_s{seed}.pth')
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*70}")
    print(f"TRAINING START - {epochs} EPOCHS")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                   leave=False, ncols=100)
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            
            optimizer.zero_grad(set_to_none=False)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                                     leave=False, ncols=100):
                images = images.to(device, non_blocking=False)
                masks = masks.to(device, non_blocking=False)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val: {avg_val_loss:.6f}", end='')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'seed': seed,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'complexity': complexity,
                'inference_stats': inference_stats,
                'config': {
                    'model': model_name,
                    'loss': loss_name,
                    'dataset': 'floodnet',
                    'batch_size': batch_size,
                    'lr': lr,
                    'size': size,
                    'num_classes': 10
                }
            }
            
            torch.save(checkpoint, save_path, _use_new_zipfile_serialization=True)
            print(" ✓ Best")
        else:
            print()
    
    print(f"\n{'='*70}")
    print("TESTING")
    print(f"{'='*70}")
    
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            preds = torch.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(masks.cpu().numpy())
    
    miou = calculate_miou(all_preds, all_labels, 10)
    dice = calculate_dice_score(all_preds, all_labels, 10)
    pixel_acc = calculate_pixel_accuracy(all_preds, all_labels)
    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS - ACCURACY METRICS")
    print(f"{'='*70}")
    print(f"Test Loss:        {avg_test_loss:.10f}")
    print(f"mIOU:             {miou:.10f}")
    print(f"Dice Score:       {dice:.10f}")
    print(f"Pixel Accuracy:   {pixel_acc:.10f}")
    print(f"Best Val Loss:    {best_val_loss:.10f}")
    print(f"\n{'='*70}")
    print("MODEL COMPLEXITY")
    print(f"{'='*70}")
    print(f"Parameters:       {complexity['total_params']:,}")
    print(f"Model Size:       {complexity['model_size_mb']:.2f} MB")
    print(f"Memory Usage:     {complexity['memory_mb']:.2f} MB")
    print(f"GFLOPs:           {complexity['gflops']:.4f}")
    print(f"\n{'='*70}")
    print("INFERENCE PERFORMANCE")
    print(f"{'='*70}")
    print(f"Avg Inference:    {inference_stats['avg_time_s']*1000:.4f} ms")
    print(f"FPS:              {inference_stats['fps']:.2f}")
    print(f"Latency:          {inference_stats['latency_ms']:.4f} ms")
    print(f"\n{'='*70}")
    print(f"Saved:            {save_path}")
    print(f"{'='*70}\n")
    
    return {
        'test_loss': avg_test_loss,
        'miou': miou,
        'dice': dice,
        'pixel_accuracy': pixel_acc,
        'best_val_loss': best_val_loss,
        'model_path': save_path,
        'complexity': complexity,
        'inference_stats': inference_stats
    }


def main():
    parser = argparse.ArgumentParser(description='FloodNet Benchmark - 100 Epochs')
    
    parser.add_argument('--floodnet_path', type=str, 
                        default='/content/drive/MyDrive/FloodNet-Supervised_v1.0',
                        help='Path to FloodNet-Supervised_v1.0 folder')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('--loss', type=str, default='ce',
                        help='Loss function (ce for CrossEntropy)')
    parser.add_argument('--size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--output_path', type=str, default='outputs_floodnet',
                        help='Output directory')
    
    args = parser.parse_args()
    
    epochs = 100
    seed = 42
    
    set_seed(seed)
    
    print("="*70)
    print("FLOODNET BENCHMARK - 100 EPOCHS")
    print("="*70)
    print(f"Dataset Path:  {args.floodnet_path}")
    print(f"Model:         {args.model}")
    print(f"Loss:          {args.loss}")
    print(f"Epochs:        {epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Image Size:    {args.size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Seed:          {seed}")
    print(f"Num Classes:   10")
    print(f"Output Path:   {args.output_path}")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(args.floodnet_path, split, f'{split}-org-img')
        label_dir = os.path.join(args.floodnet_path, split, f'{split}-label-img')
        
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"❌ Missing: {img_dir}")
        if not os.path.exists(label_dir):
            raise FileNotFoundError(f"❌ Missing: {label_dir}")
        
        n_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        n_labels = len([f for f in os.listdir(label_dir) if f.endswith('.png')])
        print(f"✓ {split}: {n_images} images, {n_labels} labels")
    
    print("="*70)
    
    result = train_floodnet(
        model_name=args.model,
        loss_name=args.loss,
        size=args.size,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_path=args.floodnet_path,
        output_path=args.output_path,
        seed=seed
    )
    
    result_file = os.path.join(args.output_path, f'{args.model}_floodnet_results.json')
    with open(result_file, 'w') as f:
        json.dump({
            'config': {
                'model': args.model,
                'loss': args.loss,
                'epochs': epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'size': args.size,
                'seed': seed
            },
            'results': {
                'test_loss': result['test_loss'],
                'miou': result['miou'],
                'dice': result['dice'],
                'pixel_accuracy': result['pixel_accuracy'],
                'best_val_loss': result['best_val_loss']
            },
            'complexity': {
                'total_params': result['complexity']['total_params'],
                'model_size_mb': result['complexity']['model_size_mb'],
                'memory_mb': result['complexity']['memory_mb'],
                'gflops': result['complexity']['gflops']
            },
            'inference_stats': {
                'fps': result['inference_stats']['fps'],
                'latency_ms': result['inference_stats']['latency_ms'],
                'avg_time_s': result['inference_stats']['avg_time_s']
            }
        }, f, indent=2)
    
    print(f"📊 Results saved to: {result_file}\n")

if __name__ == '__main__':
    main()