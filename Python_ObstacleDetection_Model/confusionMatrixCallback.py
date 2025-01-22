from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class ConfusionMatrixCallback(Callback):
    def __init__(self, X_val, y_val, result_path):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.result_path = result_path
        self.epoch_data = []

        # Criar diretório de resultados, se não existir
        os.makedirs(self.result_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        y_pred = (self.model.predict(self.X_val) > 0.5).astype("int32").flatten()
        cm = confusion_matrix(self.y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Salvar matriz de confusão como gráfico
        self._save_confusion_matrix(epoch, cm)

        # Calcular métricas e salvar na lista
        metrics = self._calculate_metrics(tn, fp, fn, tp)
        metrics["Epoch"] = epoch + 1
        metrics["Val_Loss"] = logs.get("val_loss", None)  # Adicionar val_loss
        metrics["Val_Accuracy"] = logs.get("val_accuracy", None)  # Adicionar val_accuracy (opcional)
        self.epoch_data.append(metrics)

    def on_train_end(self, logs=None):
        # Salvar métricas detalhadas em arquivo Excel
        metrics_df = pd.DataFrame(self.epoch_data)
        metrics_file = os.path.join(self.result_path, "metrics_detailed.xlsx")
        metrics_df.to_excel(metrics_file, index=False)

    def _save_confusion_matrix(self, epoch, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Positivo', 'Negativo'], yticklabels=['Positivo', 'Negativo'])
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusão - Época {epoch + 1}")
        plt.savefig(os.path.join(self.result_path, f"confusion_matrix_epoch_{epoch + 1}.png"))
        plt.close()

    def _calculate_metrics(self, tn, fp, fn, tp):
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(self.y_val, (self.model.predict(self.X_val) > 0.5).astype("int32").flatten())
        roc = roc_auc_score(self.y_val, self.model.predict(self.X_val))

        return {
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Specificity": round(specificity, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        }
