"""
Базовая ML модель для классификации суицидального риска.

Использует объединенные мультимодальные признаки для обучения
классификатора и оценки важности признаков.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    classification_report,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_path: str) -> tuple:
    """
    Загружает данные и разделяет на признаки и метки.
    
    Args:
        data_path: Путь к CSV файлу с данными
        
    Returns:
        Кортеж (X, y, groups, feature_names)
    """
    df = pd.read_csv(data_path)
    
    if 'label' not in df.columns:
        raise ValueError("В данных отсутствует колонка 'label'")
    
    exclude_cols = ['file_id', 'segment_id', 'label', 'text', 'segment_path', 'start', 'end']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df['label'].fillna(0).astype(int)
    groups = df['file_id'].values
    
    return X, y, groups, X.columns.tolist()


def train_baseline_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    model_type: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42
) -> dict:
    """
    Обучает базовую модель с групповой стратифицированной кросс-валидацией.
    
    Args:
        X: Признаки
        y: Метки
        groups: Группы для стратификации (file_id)
        model_type: Тип модели ('logistic', 'rf', 'gbm')
        n_splits: Количество фолдов для CV
        random_state: Random state для воспроизводимости
        
    Returns:
        Словарь с результатами обучения
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if model_type == 'logistic':
        model = LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        )
    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == 'gbm':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_scores_roc = cross_val_score(
        model, X_scaled, y, cv=cv, groups=groups,
        scoring='roc_auc', n_jobs=-1
    )
    
    cv_scores_pr = cross_val_score(
        model, X_scaled, y, cv=cv, groups=groups,
        scoring='average_precision', n_jobs=-1
    )
    
    y_pred_proba = cross_val_predict(
        model, X_scaled, y, cv=cv, groups=groups,
        method='predict_proba'
    )[:, 1]
    
    y_pred = cross_val_predict(
        model, X_scaled, y, cv=cv, groups=groups
    )
    
    model.fit(X_scaled, y)
    
    results = {
        'model': model,
        'scaler': scaler,
        'cv_roc_auc_mean': cv_scores_roc.mean(),
        'cv_roc_auc_std': cv_scores_roc.std(),
        'cv_pr_auc_mean': cv_scores_pr.mean(),
        'cv_pr_auc_std': cv_scores_pr.std(),
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'y_true': y.values
    }
    
    return results


def find_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray, target_precision: float = 0.9) -> float:
    """
    Находит оптимальный порог для заданной точности.
    
    Args:
        y_true: Истинные метки
        y_pred_proba: Предсказанные вероятности
        target_precision: Целевая точность
        
    Returns:
        Оптимальный порог
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    optimal_threshold = 0.5
    max_recall = 0.0
    
    for i, prec in enumerate(precision):
        if prec >= target_precision:
            if recall[i] > max_recall:
                max_recall = recall[i]
                optimal_threshold = thresholds[i] if i < len(thresholds) else 0.5
    
    return optimal_threshold


def plot_results(results: dict, output_dir: str = 'data/results') -> None:
    """
    Создает графики результатов обучения.
    
    Args:
        results: Словарь с результатами обучения
        output_dir: Директория для сохранения графиков
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y_true = results['y_true']
    y_pred_proba = results['y_pred_proba']
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    optimal_threshold = find_optimal_threshold(y_true, y_pred_proba)
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_optimal)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (threshold = {optimal_threshold:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def get_feature_importance(model, feature_names: list, top_n: int = 20) -> pd.DataFrame:
    """
    Получает важность признаков из модели.
    
    Args:
        model: Обученная модель
        feature_names: Список названий признаков
        top_n: Количество топ признаков
        
    Returns:
        DataFrame с важностью признаков
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    return importance_df


def main():
    parser = argparse.ArgumentParser(
        description='Обучение базовой ML модели'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/features/merged_features.csv',
        help='Путь к объединенному датасету'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='logistic',
        choices=['logistic', 'rf', 'gbm'],
        help='Тип модели'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Количество фолдов для CV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/models',
        help='Директория для сохранения модели'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='data/results',
        help='Директория для сохранения результатов'
    )
    
    args = parser.parse_args()
    
    print("Загрузка данных...")
    X, y, groups, feature_names = load_data(args.data)
    
    print(f"Размер данных: {X.shape}")
    print(f"Классы: {np.bincount(y)}")
    
    print(f"\nОбучение модели ({args.model_type})...")
    results = train_baseline_model(
        X, y, groups,
        model_type=args.model_type,
        n_splits=args.n_splits
    )
    
    print(f"\nРезультаты кросс-валидации:")
    print(f"ROC-AUC: {results['cv_roc_auc_mean']:.4f} ± {results['cv_roc_auc_std']:.4f}")
    print(f"PR-AUC: {results['cv_pr_auc_mean']:.4f} ± {results['cv_pr_auc_std']:.4f}")
    
    optimal_threshold = find_optimal_threshold(results['y_true'], results['y_pred_proba'])
    y_pred_optimal = (results['y_pred_proba'] >= optimal_threshold).astype(int)
    
    print(f"\nОптимальный порог (Precision >= 0.9): {optimal_threshold:.4f}")
    print(f"\nКлассификационный отчет:")
    print(classification_report(results['y_true'], y_pred_optimal))
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'baseline_{args.model_type}.pkl')
    joblib.dump({
        'model': results['model'],
        'scaler': results['scaler'],
        'feature_names': feature_names,
        'threshold': optimal_threshold
    }, model_path)
    print(f"\nМодель сохранена в {model_path}")
    
    importance_df = get_feature_importance(results['model'], feature_names)
    if not importance_df.empty:
        importance_path = os.path.join(args.results_dir, 'feature_importance.csv')
        os.makedirs(args.results_dir, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        print(f"Важность признаков сохранена в {importance_path}")
    
    plot_results(results, args.results_dir)
    print(f"\nГрафики сохранены в {args.results_dir}")


if __name__ == '__main__':
    main()

