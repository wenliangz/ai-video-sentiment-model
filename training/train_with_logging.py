import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys
from torch.utils.tensorboard import SummaryWriter

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg

# AWS SageMaker Environment Variables
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/input/data/test")
SM_TENSORBOARD_DIR = os.environ.get('SM_TENSORBOARD_DIR', "/opt/ml/output/tensorboard")

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()

def main():
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)
    
    print("Available audio backends:", torchaudio.list_audio_backends())
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )
    
    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    writer = SummaryWriter(log_dir=SM_TENSORBOARD_DIR)
    
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)
        
        # Log training metrics to TensorBoard
        print(f'Write metrics to TensorBoard:{SM_TENSORBOARD_DIR}')
        writer.add_scalar("Loss/Train", train_loss["total"], epoch)
        writer.add_scalar("Loss/Validation", val_loss["total"], epoch)
        writer.add_scalar("Metrics/Validation_Emotion_Precision", val_metrics["emotion_precision"], epoch)
        writer.add_scalar("Metrics/Validation_Emotion_Accuracy", val_metrics["emotion_accuracy"], epoch)
        writer.add_scalar("Metrics/Validation_Sentiment_Precision", val_metrics["sentiment_precision"], epoch)
        writer.add_scalar("Metrics/Validation_Sentiment_Accuracy", val_metrics["sentiment_accuracy"], epoch)
        
        # Log metrics in SageMaker format
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))
        
        if torch.cuda.is_available():
            print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        
        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
        
        writer.flush()

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    
    print(f'Write Test metrics to TensorBoard:{SM_TENSORBOARD_DIR}')
    writer.add_scalar("Loss/Test", test_loss["total"], args.epochs)
    writer.add_scalar("Metrics/Test_Emotion_Accuracy", test_metrics["emotion_accuracy"], args.epochs)
    writer.add_scalar("Metrics/Test_Sentiment_Accuracy", test_metrics["sentiment_accuracy"], args.epochs)
    writer.add_scalar("Metrics/Test_Emotion_Precision", test_metrics["emotion_precision"], args.epochs)
    writer.add_scalar("Metrics/Test_Sentiment_Precision", test_metrics["sentiment_precision"], args.epochs)
    
    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
        ]
    }))
    
    #close TensorBoard writer
    writer.close()
    
if __name__ == "__main__":
    main()
