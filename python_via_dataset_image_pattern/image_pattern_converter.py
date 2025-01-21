import os
from PIL import Image
import shutil
import time


def process_images(input_dir, output_dir_correct, output_dir_resized, output_dir_small):
    # Criar os diretórios de saída, se não existirem
    os.makedirs(output_dir_correct, exist_ok=True)
    os.makedirs(output_dir_resized, exist_ok=True)
    os.makedirs(output_dir_small, exist_ok=True)

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        try:
            # Abrir a imagem
            with Image.open(filepath) as img:
                width, height = img.size

                if width == 750 and height == 1000:
                    # Copiar imagens com dimensões corretas
                    output_path = os.path.join(output_dir_correct, filename)
                    shutil.copy(filepath, output_path)
                    print(f"A imagem {filename} foi copiada para {output_dir_correct}.")
                elif width > 650 and height > 800:
                    # Redimensionar imagens maiores
                    img_resized = img.resize((750, 1000), Image.Resampling.LANCZOS)
                    output_path = os.path.join(output_dir_resized, filename)
                    img_resized.save(output_path)
                    print(f"A imagem {filename} foi redimensionada e salva em {output_dir_resized}.")
                else:
                    # Copiar imagens menores
                    output_path = os.path.join(output_dir_small, filename)
                    shutil.copy(filepath, output_path)
                    print(f"A imagem {filename} foi copiada para {output_dir_small}.")

            # Esperar brevemente antes de remover o arquivo
            time.sleep(0.1)

            # Apagar o arquivo do diretório original
            os.remove(filepath)
            print(f"A imagem {filename} foi removida do diretório original.")

        except Exception as e:
            print(f"Erro ao processar a imagem {filename}: {e}")


# Diretórios
input_directory = "/Projetos/2024_Phd_ObstacleDetectionModel/via_clear"
output_directory_correct = "/Projetos/2024_Phd_ObstacleDetectionModel/via_clear/correct"
output_directory_resized = "/Projetos/2024_Phd_ObstacleDetectionModel/via_clear/resized"
output_directory_small = "/Projetos/2024_Phd_ObstacleDetectionModel/via_clear/small"

# Executar o processamento
process_images(input_directory, output_directory_correct, output_directory_resized, output_directory_small)
