import os
from PIL import Image
import pillow_heif  # Biblioteca para ler imagens HEIC
import shutil
import time


def convert_heic_to_jpg(heic_path, jpg_path):
    """Converte uma imagem HEIC para JPG."""
    try:
        heif_image = pillow_heif.open_heif(heic_path)  # Abre a imagem HEIC
        img = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)  # Converte para PIL
        img = img.convert("RGB")  # Converte para RGB
        img.save(jpg_path, "JPEG", quality=95)  # Salva como JPG
        return jpg_path
    except Exception as e:
        print(f"Erro ao converter {heic_path}: {e}")
        return None


def process_images(input_dir, output_dir_correct, output_dir_resized, output_dir_small):
    """Processa imagens HEIC, convertendo para JPG e redimensionando conforme necessário."""

    os.makedirs(output_dir_correct, exist_ok=True)
    os.makedirs(output_dir_resized, exist_ok=True)
    os.makedirs(output_dir_small, exist_ok=True)

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        if not filename.lower().endswith(".heic"):
            print(f"⚠️ Ignorando {filename}, pois não é um arquivo HEIC.")
            continue

        try:
            # Definir caminho de saída para conversão
            jpg_filename = filename.rsplit(".", 1)[0] + ".jpg"
            jpg_path = os.path.join(input_dir, jpg_filename)

            # Converter HEIC para JPG
            converted_path = convert_heic_to_jpg(filepath, jpg_path)

            if converted_path:
                with Image.open(converted_path) as img:
                    width, height = img.size

                    if width == 750 and height == 1000:
                        shutil.move(converted_path, os.path.join(output_dir_correct, jpg_filename))
                        print(f"✅ {filename} convertida e copiada para {output_dir_correct}.")
                    elif width > 650 and height > 800:
                        img_resized = img.resize((750, 1000), Image.Resampling.LANCZOS)
                        img_resized.save(os.path.join(output_dir_resized, jpg_filename), "JPEG", quality=95)
                        print(f"🔄 {filename} redimensionada e salva em {output_dir_resized}.")
                    else:
                        shutil.move(converted_path, os.path.join(output_dir_small, jpg_filename))
                        print(f"📉 {filename} convertida e copiada para {output_dir_small}.")

                time.sleep(0.1)
                os.remove(filepath)  # Remove o arquivo HEIC original
                print(f"🗑️ {filename} removida do diretório original.")

        except Exception as e:
            print(f"❌ Erro ao processar {filename}: {e}")


# Diretórios
input_directory = "/home/jeffersonpasserini/projetos/2024_Phd_ObstacleDetectionModel/imagens"
output_directory_correct = "/home/jeffersonpasserini/projetos/2024_Phd_ObstacleDetectionModel/correct"
output_directory_resized = "/home/jeffersonpasserini/projetos/2024_Phd_ObstacleDetectionModel/resized"
output_directory_small = "/home/jeffersonpasserini/projetos/2024_Phd_ObstacleDetectionModel/small"

# Executar o processamento
process_images(input_directory, output_directory_correct, output_directory_resized, output_directory_small)
