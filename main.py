import numpy as np
import pandas as pd
import torch
import random
from typing import List, Dict
from dataclasses import dataclass
from pathlib import Path

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# Global Configuration


@dataclass
class Config:
    SEED: int = 42
    CATEGORIES: List[str] = ("Alimentação", "Transporte", "Lazer")
    MAX_LENGTH: int = 64
    MODEL_NAME: str = "distilbert-base-multilingual-cased"
    OUTPUT_DIR: Path = Path("./results")
    MODEL_VERSION: str = "v6"


config = Config()
CAT_MAPPING: Dict[str, int] = {
    cat: idx for idx, cat in enumerate(config.CATEGORIES)}


def set_seed(seed: int = config.SEED) -> None:
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DataAugmentation:
    @staticmethod
    def text_variations(text: str) -> List[str]:
        """Generate text variations for data augmentation"""
        return [
            text,
            text.lower(),
            text.upper(),
            text.capitalize(),
            f"Minha {text}",
            f"Última {text}",
            f"Próxima {text}",
            text.replace('a', '@').replace('o', '0')
        ]


def gerar_dados_sinteticos(qtd: int = 5000) -> pd.DataFrame:
    """Generate synthetic data with balanced class distribution"""
    estrategias = {
        "Alimentação": [
            "Almoço no restaurante", "Compra no supermercado",
            "Refeição rápida", "Jantar em casa", "Lanche no trabalho"
        ],
        "Transporte": [
            "Uber para o trabalho", "Combustível do carro",
            "Manutenção do veículo", "Estacionamento", "Pedágio"
        ],
        "Lazer": [
            "Ingresso de cinema", "Assinatura de streaming",
            "Show de música", "Parque de diversões", "Viagem de fim de semana"
        ]
    }

    dados = []
    samples_per_category = qtd // len(estrategias)

    for categoria, exemplos in estrategias.items():
        for _ in range(samples_per_category):
            exemplo = random.choice(exemplos)
            variacoes = DataAugmentation.text_variations(exemplo)
            variacao = random.choice(variacoes)
            dados.append({"descricao": variacao, "categoria": categoria})

    return pd.DataFrame(dados)


class TransacoesDataset(Dataset):
    """Custom Dataset for financial transactions"""
    def __init__(self, encodings: Dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def compute_metrics_apresentacao(pred) -> Dict:
    """Compute and visualize model metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    metricas = {
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1_score': f1_score(labels, preds, average='weighted'),
        'accuracy': accuracy_score(labels, preds),
        'kappa': cohen_kappa_score(labels, preds)
    }

    plot_metrics(labels, preds, metricas)
    return metricas


def plot_metrics(
        labels: np.ndarray, preds: np.ndarray, metricas: Dict) -> None:
    """Plot performance metrics"""
    plt.figure(figsize=(20, 6))

    # Confusion Matrix
    plt.subplot(131)
    conf_matrix = confusion_matrix(labels, preds)
    sns.heatmap(conf_matrix, annot=True, fmt='d',
                cmap='viridis',
                xticklabels=config.CATEGORIES,
                yticklabels=config.CATEGORIES)
    plt.title('Detailed Confusion Matrix')

    # Per-category Metrics
    plt.subplot(132)
    report = classification_report(
        labels, preds,
        target_names=config.CATEGORIES,
        output_dict=True
    )
    categories = list(report.keys())[:-3]
    precisions = [report[cat]['precision'] for cat in categories]
    plt.bar(categories, precisions)
    plt.title('Precision by Category')
    plt.xticks(rotation=45)

    # Performance Curve
    plt.subplot(133)
    plt.plot(list(metricas.keys()), list(metricas.values()), marker='o')
    plt.title('Performance Metrics')
    plt.xticks(rotation=45)

    plt.tight_layout()
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(config.OUTPUT_DIR / 'model_metrics.png')
    plt.close()


def get_training_args() -> TrainingArguments:
    """Get training arguments with optimal hyperparameters"""
    return TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=15,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1_score',
        fp16=True,
        gradient_accumulation_steps=2,
        report_to="tensorboard"
    )


def main() -> None:
    set_seed()

    # Data preparation
    df = gerar_dados_sinteticos(5000)
    df['label'] = df['categoria'].map(CAT_MAPPING)

    # Train-test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['descricao'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        stratify=df['label'],
        random_state=config.SEED
    )

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.CATEGORIES)
    )

    # Prepare datasets
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=config.MAX_LENGTH
    )
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=config.MAX_LENGTH
    )

    train_dataset = TransacoesDataset(train_encodings, train_labels)
    val_dataset = TransacoesDataset(val_encodings, val_labels)

    # Training
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_apresentacao,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    # Save model and tokenizer
    model_path = config.OUTPUT_DIR / (
        f'modelo_financeiro_{config.MODEL_VERSION}'
    )
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()
