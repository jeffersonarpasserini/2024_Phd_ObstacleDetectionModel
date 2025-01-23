import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.utils import plot_model

# Caminho base do projeto
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Definição dos caminhos
MODEL_DIR = os.path.join(BASE_PATH, 'model')
COMBINED_MODEL_DIR = os.path.join(MODEL_DIR, 'combined_model')
#TFLITE_MODEL_COMBINED_DIR = os.path.join(COMBINED_MODEL_DIR, 'tflite')

# Garantir que os diretórios existem
os.makedirs(COMBINED_MODEL_DIR, exist_ok=True)
#os.makedirs(TFLITE_MODEL_COMBINED_DIR, exist_ok=True)

# Renomeia as camadas de um modelo adicionando um prefixo.
def rename_layers(model, prefix):
    for layer in model.layers:
        layer._name = prefix + layer.name
    return model

# Gera um modelo combinado usando MobileNetV2 como extrator de features e um modelo denso pré-treinado.
def generateCombinedModel(split_index=0):

    try:
        print('Generating Combined Model')

        file_name_h5 = f"{split_index:02d}_classifier_model.h5"
        DENSE_MODEL_PATH = os.path.join(MODEL_DIR, 'classifier_model', file_name_h5)

        file_name_h5_combined = f"{split_index:02d}_classifier_model_combined.h5"
        COMBINED_MODEL_PATH = os.path.join(COMBINED_MODEL_DIR, file_name_h5_combined)

        file_name_model_summary = f"{split_index:02d}_model_summary_combined.csv"
        COMBINED_MODEL_SUMMARY_PATH = os.path.join(COMBINED_MODEL_DIR, file_name_model_summary)

        file_name_tflite_combined = f"{split_index:02d}_classifier_model_combined.tflite"
        TFLITE_MODEL_COMBINED_PATH = os.path.join(COMBINED_MODEL_DIR, file_name_tflite_combined)


        # Carregar o modelo denso já treinado
        if not os.path.exists(DENSE_MODEL_PATH):
            raise FileNotFoundError(f"Modelo denso não encontrado: {DENSE_MODEL_PATH}")

        model_dense = load_model(DENSE_MODEL_PATH)
        model_dense = rename_layers(model_dense, 'dense_')
        print('Classifier imported - h5 file')

        # Configurações para o modelo MobileNetV2
        IMAGE_CHANNELS = 3
        image_size = (224, 224)
        input_image = Input(shape=(224, 224, 3))

        # Carregar o MobileNetV2 pré-treinado
        mobilenetv2_base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                                       input_shape=image_size + (IMAGE_CHANNELS,))
        mobilenetv2_base = rename_layers(mobilenetv2_base, 'mobilenetv2_')
        print('MobileNetV2 base layers imported - imagenet - extract features')

        # Congelar as camadas do MobileNetV2
        for layer in mobilenetv2_base.layers:
            layer.trainable = False

        # Combinar MobileNetV2 com o modelo denso
        mobilenetv2_features = mobilenetv2_base(input_image)
        flatten_mobilenetv2 = Flatten()(mobilenetv2_features)
        dense_output = model_dense(flatten_mobilenetv2)
        model_combined = Model(inputs=input_image, outputs=dense_output)

        # Compilar o modelo
        model_combined.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        #gera sumario do modelo combinado
        layer_info = [
            {
                "Name": layer.name,
                "Type": type(layer).__name__,  # Tipo da camada
                "Output Shape": layer.output_shape if hasattr(layer, 'output_shape') else None,  # Formato de saída
                "Params": layer.count_params() if hasattr(layer, 'count_params') else 0  # Número de parâmetros
            }
            for layer in model_combined.layers
        ]
        pd.DataFrame(layer_info).to_csv(COMBINED_MODEL_SUMMARY_PATH, index=False)
        print('Combined model built - MobileNetV2 base Layers+Classifier Layers')

        # Salvar o modelo combinado em formato Keras
        model_combined.save(COMBINED_MODEL_PATH)
        print(f"Modelo combinado salvo em: {COMBINED_MODEL_PATH}")

        # Visualizar a arquitetura do modelo (opcional)
        file_model_combined_architecture = f"{split_index:02d}_model_combined_architecture.png"
        plot_model(model_combined, to_file=os.path.join(COMBINED_MODEL_DIR, file_model_combined_architecture), show_shapes=True)

        # Converter o modelo para TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model_combined)
        tflite_model = converter.convert()

        # Salvar o modelo TensorFlow Lite
        with open(TFLITE_MODEL_COMBINED_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"Modelo combinado TensorFlow Lite salvo em: {TFLITE_MODEL_COMBINED_PATH}")

    except Exception as e:
        print(f"Erro ao gerar o modelo combinado: {str(e)}")


if __name__ == "__main__":
    generateCombinedModel()
