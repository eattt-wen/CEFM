import os
import time
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import yaml

from data.dataset import MelanomaDataset
from models.feature_extractor import load_image_model, ImageFeatureExtractor
from models.projection import ContrastiveLearningModel
from utils.losses import nt_xent_loss
from utils.io import mkdir_if_not_exists

def train_projection_heads(model, train_loader, val_loader, optimizer, epochs, device, output_dir, patience=5):
    model.train()
    history = {'train_loss': [], 'val_loss': [], 'step': []}

    best_val_loss = float('inf')
    no_improve_epochs = 0
    log_file = open(os.path.join(output_dir, 'projection_heads_training.txt'), 'w')
    step_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (images, clinical_features, _) in enumerate(train_loader):
            images = images.to(device)
            clinical_features = clinical_features.to(device)

            optimizer.zero_grad()
            img_proj, clin_proj = model(images, clinical_features)
            loss = nt_xent_loss(img_proj, clin_proj, model.temperature)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                step_counter += 1
                avg_loss = running_loss / (batch_idx + 1)
                msg = f'epoch {epoch+1}/{epochs}, batch {batch_idx}/{len(train_loader)}, loss {loss.item():.4f}, avg {avg_loss:.4f}'
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()

        # 验证
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, clinical_features, _ in val_loader:
                images = images.to(device)
                clinical_features = clinical_features.to(device)
                img_proj, clin_proj = model(images, clinical_features)
                val_loss += nt_xent_loss(img_proj, clin_proj, model.temperature).item()
        val_loss /= len(val_loader)

        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['step'].append(epoch)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': history,
        }, os.path.join(output_dir, f'projection_heads_epoch_{epoch+1}.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'projection_heads_best.pth'))
            msg = f'** New best model saved (val_loss={val_loss:.4f})'
        else:
            no_improve_epochs += 1
            msg = f'epoch {epoch+1} done (train_loss={train_loss:.4f}, val_loss={val_loss:.4f}) — {no_improve_epochs} epochs no improve'
        print(msg)
        log_file.write(msg + '\n\n')
        log_file.flush()

        if no_improve_epochs >= patience:
            stop_msg = f'Early stopping after {epoch+1} epochs'
            print(stop_msg)
            log_file.write(stop_msg + '\n')
            break

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['train_loss'], label='Train Loss')
    plt.plot(history['step'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'projection_heads_loss.png'))
    plt.close()

    return history

def train_projection_head_stage(cfg):
    torch.manual_seed(cfg['training']['seed'])
    np.random.seed(cfg['training']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir = cfg['data']['image_dir']
    train_csv = cfg['data']['train_csv']
    test_csv = cfg['data']['test_csv']
    image_model_path = cfg['model']['image_model_path']
    output_dir = cfg['model']['output_dir']

    mkdir_if_not_exists(output_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    full_ds = MelanomaDataset(image_dir, train_csv, transform)
    test_ds = MelanomaDataset(image_dir, test_csv, transform)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)

    vit = load_image_model(image_model_path)
    feat_extractor = ImageFeatureExtractor(vit)
    dummy = torch.randn(1,3,224,224)
    with torch.no_grad():
        img_dim = feat_extractor(dummy).size(1)
    clin_dim = len(full_ds.clinical_features[0])

    model = ContrastiveLearningModel(
        image_feature_extractor=feat_extractor,
        image_feature_dim=img_dim,
        clinical_feature_dim=clin_dim,
        projection_dim=cfg['training'].get('projection_dim',128),
        temperature=cfg['training']['temperature']
    ).to(device)

    for p in model.parameters(): p.requires_grad = False
    for m in [model.image_projector, model.clinical_projector]:
        for p in m.parameters(): p.requires_grad = True
    wd = cfg['training']['weight_decay']
    lr = cfg['training']['lr']
    wd = float(wd)
    lr = float(lr)
    optimizer = optim.Adam(
        [
            {'params': model.image_projector.parameters(), 'lr': lr},
            {'params': model.clinical_projector.parameters(), 'lr': lr},
        ],
        weight_decay=wd
    )

    history = train_projection_heads(
        model, train_loader, val_loader, optimizer,
        cfg['training']['epochs_contrastive'], device,
        output_dir, cfg['training']['patience']
    )

    torch.save(model.state_dict(), os.path.join(output_dir, 'projection_heads_final.pth'))
    return model, feat_extractor, img_dim, clin_dim

if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml', 'r'))
    train_projection_head_stage(cfg)
