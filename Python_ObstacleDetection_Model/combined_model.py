import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Input
from tensorflow.keras.utils import plot_model

# Caminho base do projeto
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Definição dos caminhos
MODEL_DIR = os.path.join(BASE_PATH, 'model')
COMBINED_MODEL_DIR = os.path.join(MODEL_DIR, 'combined_model')
TFLITE_MODEL_COMBINED_DIR = os.path.join(COMBINED_MODEL_DIR, 'tflite')
DENSE_MODEL_PATH = os.path.join(MODEL_DIR, 'classifier_model', 'classifier_model.h5')
COMBINED_MODEL_PATH = os.path.join(COMBINED_MODEL_DIR, 'model_combined_image_input_mobilenetv2.h5')
TFLITE_MODEL_COMBINED_PATH = os.path.join(TFLITE_MODEL_COMBINED_DIR, 'model_combined.tflite')

# Garantir que os diretórios existem
os.makedirs(COMBINED_MODEL_DIR, exist_ok=True)
os.makedirs(TFLITE_MODEL_COMBINED_DIR, exist_ok=True)

# Renomeia as camadas de um modelo adicionando um prefixo.
def rename_layers(model, prefix):
    for layer in model.layers:
        layer._name = prefix + layer.name
    return model

# Gera um modelo combinado usando MobileNetV2 como extrator de features e um modelo denso pré-treinado.
def generateCombinedModel():

    try:
        # Carregar o modelo denso já treinado
        if not os.path.exists(DENSE_MODEL_PATH):
            raise FileNotFoundError(f"Modelo denso não encontrado: {DENSE_MODEL_PATH}")

        model_dense = load_model(DENSE_MODEL_PATH)
        model_dense = rename_layers(model_dense, 'dense_')

        # Configurações para o modelo MobileNetV2
        IMAGE_CHANNELS = 3
        image_size = (224, 224)
        input_image = Input(shape=(224, 224, 3))

        # Carregar o MobileNetV2 pré-treinado
        mobilenetv2_base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                                       input_shape=image_size + (IMAGE_CHANNELS,))
        mobilenetv2_base = rename_layers(mobilenetv2_base, 'mobilenetv2_')

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

        # Salvar o modelo combinado em formato Keras
        model_combined.save(COMBINED_MODEL_PATH)
        print(f"Modelo combinado salvo em: {COMBINED_MODEL_PATH}")

        # Visualizar a arquitetura do modelo (opcional)
        plot_model(model_combined, to_file=os.path.join(COMBINED_MODEL_DIR, 'model_architecture.png'), show_shapes=True)

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
