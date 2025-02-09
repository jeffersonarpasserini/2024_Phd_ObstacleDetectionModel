from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class ConfusionMatrixCallback(Callback):
    def __init__(self, X_val, y_val, result_path, split_index=0):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.result_path = result_path
        self.split_index = split_index
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
        metrics.update({
            "Split_Index": self.split_index,  # Adicionar o índice do split
            "Epoch": epoch + 1,
            "Training_Accuracy": metrics.get("Accuracy"),
            "Training_Val_Accuracy": logs.get("val_accuracy", None),
            "Training_Loss": logs.get("loss", None),
            "Training_Val_Loss": logs.get("val_loss", None),
            "Training_F1": metrics.get("F1-Score"),
            "Training_Val_F1": logs.get("val_f1", None),
            "Training_AUC": metrics.get("ROC-AUC"),
            "Training_Val_AUC": logs.get("val_auc", None),
            "Training_Precision": metrics.get("Precision"),
            "Training_Val_Precision": logs.get("val_precision", None),
            "Training_Recall": metrics.get("Recall"),
            "Training_Val_Recall": logs.get("val_recall", None),
            "Training_Specificity": metrics.get("Specificity"),
            "Training_Val_Specificity": logs.get("val_specificity", None),
            "Training_TN": tn,
            "Training_FP": fp,
            "Training_FN": fn,
            "Training_TP": tp,
        })
        self.epoch_data.append(metrics)

    def on_train_end(self, logs=None):
        # Salvar métricas detalhadas em arquivo Excel
        metrics_df = pd.DataFrame(self.epoch_data)
        metrics_file = os.path.join(self.result_path, f"{self.split_index:02d}_results.classifier_training.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Métricas de treinamento do modelo salvas em {metrics_file}")

    def _save_confusion_matrix(self, epoch, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Positivo', 'Negativo'], yticklabels=['Positivo', 'Negativo'])
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.title(f"Matriz de Confusão - Época {epoch + 1}")
        plt.savefig(os.path.join(self.result_path, f"Split_{self.split_index:02d}_Epoch_{epoch+1}_confusion_matrix.png"))
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
