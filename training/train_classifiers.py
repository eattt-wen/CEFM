import os
import yaml
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from data.dataset import MelanomaDataset
from models.feature_extractor import load_image_model, ImageFeatureExtractor
from models.projection import load_projection_model
from models.classifiers import ImageClassifier, ClinicalClassifier
from evaluation.evaluate import evaluate_classifier
from utils.io import mkdir_if_not_exists


def train_classifier(model, train_loader, criterion, optimizer, epochs, device, output_dir, model_name):
    model.train()
    history = {'loss': [], 'acc': [], 'step': []}
    log_file = open(os.path.join(output_dir, f'{model_name}_training.txt'), 'w')
    step_counter = 0

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (images, clinical_features, labels) in enumerate(train_loader):
            inputs = images.to(device) if model_name == 'image_classifier' else clinical_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                step_counter += 1
                avg_loss = running_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                history['loss'].append(avg_loss)
                history['acc'].append(acc)
                history['step'].append(step_counter)
                msg = f'epoch {epoch+1}/{epochs}, batch {batch_idx}/{len(train_loader)}, loss {loss.item():.4f}, acc {acc:.2f}%'
                print(msg)
                log_file.write(msg + '\n')
                log_file.flush()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader),
            'accuracy': 100.0 * correct / total,
            'history': history
        }, os.path.join(output_dir, f'{model_name}_epoch_{epoch+1}.pth'))

        msg = f'epoch {epoch+1} done, loss {running_loss/len(train_loader):.4f}, acc {100.0*correct/total:.2f}%'
        print(msg)
        log_file.write(msg + '\n\n')
        log_file.flush()

    return history


def train_classifiers_stage(cfg):
    seed = int(cfg['training']['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_size   = int(cfg['training']['batch_size'])
    lr           = float(cfg['training']['lr'])
    weight_decay = float(cfg['training']['weight_decay'])
    epochs       = int(cfg['training']['epochs_classifier'])

    image_dir = cfg['data']['image_dir']
    train_csv = cfg['data']['train_csv']
    test_csv  = cfg['data']['test_csv']
    image_model_path = cfg['model']['image_model_path']
    output_dir = cfg['model']['output_dir']
    mkdir_if_not_exists(output_dir)

    projection_model_path = cfg['model'].get(
        'projection_model_path',
        os.path.join(output_dir, 'projection_heads_final.pth')
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = MelanomaDataset(image_dir, train_csv, transform)
    test_ds  = MelanomaDataset(image_dir, test_csv, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

    vit = load_image_model(image_model_path)
    feat_extractor = ImageFeatureExtractor(vit).to(device)

    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        img_dim = feat_extractor(dummy).size(1)
    clin_dim = len(train_ds.clinical_features[0])

    proj = load_projection_model(
        projection_model_path,
        feat_extractor,
        img_dim,
        clin_dim
    ).to(device)

    img_clf = ImageClassifier(feat_extractor, proj.image_projector).to(device)
    clin_clf = ClinicalClassifier(proj.clinical_projector).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    img_opt  = optim.Adam(img_clf.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    clin_opt = optim.Adam(clin_clf.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    train_classifier(img_clf, train_loader, criterion, img_opt,
                     epochs, device, output_dir, 'image_classifier')
    train_classifier(clin_clf, train_loader, criterion, clin_opt,
                     epochs, device, output_dir, 'clinical_classifier')

    torch.save(img_clf.state_dict(),  os.path.join(output_dir, 'image_classifier_final.pth'))
    torch.save(clin_clf.state_dict(), os.path.join(output_dir, 'clinical_classifier_final.pth'))

    evaluate_classifier(img_clf, test_loader, device, 'image_classifier', output_dir)
    evaluate_classifier(clin_clf, test_loader, device, 'clinical_classifier', output_dir)


if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/default.yaml', 'r'))
    train_classifiers_stage(cfg)
