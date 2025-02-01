import numpy as np
import pandas as pd
import torch
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Bibliotecas para Machine Learning e visualização
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, cohen_kappa_score, classification_report,
    confusion_matrix
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

# --------------------------
# Configuração Global
# --------------------------
@dataclass
class Config:
    SEED: int = 42
    CATEGORIES: List[str] = (
        "Alimentação", "Transporte", "Lazer", "Educação", "Saúde",
        "Vestuário", "Moradia", "Serviços", "Entretenimento", "Tecnologia"
    )
    MAX_LENGTH: int = 64
    MODEL_NAME: str = "distilbert-base-multilingual-cased"
    OUTPUT_DIR: Path = Path("./results")
    MODEL_VERSION: str = "v9"
    SYNTHETIC_DATA_SAMPLES: int = 10000  # Número total de amostras sintéticas

config = Config()
CAT_MAPPING: Dict[str, int] = {cat: idx for idx, cat in enumerate(config.CATEGORIES)}

def set_seed(seed: int = config.SEED) -> None:
    """Define as sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --------------------------
# Data Augmentation e Geração de Dados
# --------------------------
class DataAugmentation:
    @staticmethod
    def text_variations(text: str) -> List[str]:
        """Gera variações textuais para aumentar os dados de treinamento."""
        variations = {
            text,
            text.lower(),
            text.upper(),
            text.capitalize(),
            f"Minha {text}",
            f"Última {text}",
            f"Próxima {text}",
            text.replace('a', '@').replace('o', '0'),
            f"{text} hoje",
            f"{text} agora"
        }
        return list(variations)

# Exemplos expandidos para cada categoria
CATEGORY_EXAMPLES: Dict[str, List[str]] = {
    "Alimentação": [
        "Almoço no restaurante", "Compra no supermercado",
        "Refeição rápida", "Jantar em casa", "Café da manhã", "Lanche da tarde",
        "Delivery de comida", "Jantar com amigos"
    ],
    "Transporte": [
        "Uber para o trabalho", "Combustível do carro", "Manutenção do veículo",
        "Estacionamento", "Passagem de ônibus", "Táxi noturno", "Bilhete de metrô"
    ],
    "Lazer": [
        "Ingresso de cinema", "Assinatura de streaming",
        "Show de música", "Parque de diversões", "Passeio no parque", "Aluguel de bicicleta"
    ],
    "Educação": [
        "Mensalidade escolar", "Curso online", "Compra de livros",
        "Material escolar", "Workshop profissional", "Inscrição em seminário"
    ],
    "Saúde": [
        "Consulta médica", "Compra de remédios", "Exame laboratorial",
        "Sessão de fisioterapia", "Vacinação", "Internação hospitalar"
    ],
    "Vestuário": [
        "Compra de roupas", "Acessórios de moda", "Sapatos", "Loja de vestuário",
        "Promoção de roupas", "Nova coleção de inverno"
    ],
    "Moradia": [
        "Aluguel de imóvel", "Conta de luz", "Conta de água",
        "Manutenção residencial", "Compra de móveis", "Serviços de limpeza"
    ],
    "Serviços": [
        "Assinatura de internet", "Serviço de streaming", "Consultoria",
        "Serviço de entrega", "Assinatura de revista", "Limpeza residencial"
    ],
    "Entretenimento": [
        "Show ao vivo", "Festival de música", "Teatro", "Evento esportivo",
        "Parque temático", "Passeio cultural"
    ],
    "Tecnologia": [
        "Compra de smartphone", "Assinatura de software", "Compra de notebook",
        "Acessórios de computador", "Upgrade de dispositivo", "Serviços de nuvem"
    ]
}

def gerar_dados_sinteticos(qtd: int = config.SYNTHETIC_DATA_SAMPLES) -> pd.DataFrame:
    """
    Gera dados sintéticos de transações com distribuição balanceada entre as categorias.
    """
    dados: List[Dict[str, Any]] = []
    num_categories = len(CATEGORY_EXAMPLES)
    samples_per_category = qtd // num_categories

    for categoria, exemplos in CATEGORY_EXAMPLES.items():
        for _ in range(samples_per_category):
            base_example = random.choice(exemplos)
            variations = DataAugmentation.text_variations(base_example)
            exemplo_variado = random.choice(variations)
            dados.append({"descricao": exemplo_variado, "categoria": categoria})
    
    # Se qtd não for divisível exatamente, adiciona amostras extras de categorias aleatórias.
    while len(dados) < qtd:
        categoria = random.choice(list(CATEGORY_EXAMPLES.keys()))
        base_example = random.choice(CATEGORY_EXAMPLES[categoria])
        variations = DataAugmentation.text_variations(base_example)
        exemplo_variado = random.choice(variations)
        dados.append({"descricao": exemplo_variado, "categoria": categoria})
        
    return pd.DataFrame(dados)

# --------------------------
# Dataset Customizado
# --------------------------
class TransacoesDataset(Dataset):
    """Dataset customizado para transações financeiras."""
    def __init__(self, encodings: Dict[str, Any], labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)

# --------------------------
# Funções de Visualização
# --------------------------
def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, categories: List[str], save_path: Path) -> None:
    """
    Plota e salva a matriz de confusão.
    """
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
                xticklabels=categories, yticklabels=categories)
    plt.title("Matriz de Confusão")
    plt.xlabel("Rótulo Predito")
    plt.ylabel("Rótulo Real")
    plt.tight_layout()
    save_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path / "confusion_matrix.png")
    plt.close()

def plot_precision_per_category(labels: np.ndarray, preds: np.ndarray, categories: List[str], save_path: Path) -> None:
    """
    Plota e salva a precisão por categoria em um gráfico de barras.
    """
    report = classification_report(labels, preds, target_names=categories, output_dict=True, zero_division=0)
    precisions = [report[cat]['precision'] for cat in categories if cat in report]
    
    plt.figure(figsize=(8, 6))
    plt.bar(categories[:len(precisions)], precisions, color='skyblue')
    plt.title("Precisão por Categoria")
    plt.xlabel("Categoria")
    plt.ylabel("Precisão")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path / "precision_per_category.png")
    plt.close()

def plot_overall_metrics(metrics: dict, save_path: Path) -> None:
    """
    Plota e salva as métricas de desempenho global.
    """
    plt.figure(figsize=(8, 6))
    keys = list(metrics.keys())
    values = list(metrics.values())
    plt.plot(keys, values, marker='o', linestyle='-', color='green')
    plt.title("Métricas Globais de Desempenho")
    plt.xlabel("Métrica")
    plt.ylabel("Valor")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path / "overall_performance_metrics.png")
    plt.close()

def plot_all_metrics(labels: np.ndarray, preds: np.ndarray, metrics: dict, categories: List[str], output_dir: Path) -> None:
    """
    Gera e salva todos os gráficos de avaliação.
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_confusion_matrix(labels, preds, categories, output_dir)
    plot_precision_per_category(labels, preds, categories, output_dir)
    plot_overall_metrics(metrics, output_dir)

# --------------------------
# Função de Cálculo das Métricas
# --------------------------
def compute_metrics_apresentacao(pred) -> dict:
    """
    Calcula as métricas de avaliação e gera os gráficos correspondentes.
    
    Retorna:
        Um dicionário com as métricas calculadas.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    metrics = {
        'precision': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1_score': f1_score(labels, preds, average='weighted', zero_division=0),
        'accuracy': accuracy_score(labels, preds),
        'kappa': cohen_kappa_score(labels, preds)
    }
    
    # Gera e salva os gráficos de avaliação
    plot_all_metrics(labels, preds, metrics, config.CATEGORIES, config.OUTPUT_DIR)
    
    return metrics

# --------------------------
# Parâmetros de Treinamento com Ajustes para Reduzir Overfitting
# --------------------------
def get_training_args() -> TrainingArguments:
    """
    Retorna os argumentos de treinamento com hiperparâmetros ajustados.
    Foram reduzidas as épocas para evitar sobreajuste.
    """
    return TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=3,  # Redução do número de épocas para evitar overfitting
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1_score',
        fp16=True,
        gradient_accumulation_steps=2,
        report_to="tensorboard"
    )

# --------------------------
# Pipeline Principal de Treinamento
# --------------------------
def main() -> None:
    set_seed()
    
    # Preparação dos dados
    df = gerar_dados_sinteticos(config.SYNTHETIC_DATA_SAMPLES)
    df['label'] = df['categoria'].map(CAT_MAPPING)
    
    # Divisão entre treinamento e validação com estratificação
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['descricao'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        stratify=df['label'],
        random_state=config.SEED
    )
    
    # Inicialização do tokenizer e do modelo
    tokenizer = DistilBertTokenizer.from_pretrained(config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.CATEGORIES)
    )
    # Ajuste de dropout para reduzir overfitting
    model.config.attention_dropout = 0.3
    model.config.hidden_dropout_prob = 0.3
    
    # Tokenização dos textos
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
    
    # Criação dos datasets
    train_dataset = TransacoesDataset(train_encodings, train_labels)
    val_dataset = TransacoesDataset(val_encodings, val_labels)
    
    # Configuração do Trainer com callback de early stopping
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_apresentacao,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    trainer.train()
    
    # Salvando o modelo e o tokenizer
    model_path = config.OUTPUT_DIR / f'modelo_financeiro_{config.MODEL_VERSION}'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    main()
