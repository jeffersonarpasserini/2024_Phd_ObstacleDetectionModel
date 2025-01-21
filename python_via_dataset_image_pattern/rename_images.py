import os
import re

def rename_images(directory, start_number=0, clear_nonclear="nonclear"):
    # Lista todos os arquivos no diretório especificado
    files = sorted([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    current_number = start_number

    for filename in files:
        # Extensão do arquivo
        file_extension = os.path.splitext(filename)[1]
        # Novo nome no formato desejado
        if (clear_nonclear == "clear"):
            new_name = f"clear.{current_number:03d}{file_extension}"
        else:
            new_name = f"nonclear.{current_number:03d}{file_extension}"

        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"Renomeado: {filename} -> {new_name}")
        except Exception as e:
            print(f"Erro ao renomear {filename}: {e}")

        current_number += 1


def add_leading_zero(directory):
    # Regex para capturar arquivos no padrão nonclear.### ou clear.###
    pattern = re.compile(r"(nonclear|clear)\.(\d+)(\.\w+)")

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extrai o prefixo, número atual e a extensão do arquivo
            prefix = match.group(1)
            current_number = match.group(2)
            file_extension = match.group(3)

            # Ignorar arquivos que já possuem quatro dígitos
            if len(current_number) == 4:
                print(f"Ignorado: {filename} já possui quatro dígitos.")
                continue

            # Adiciona um zero à esquerda ao número
            new_number = f"{int(current_number):04d}"
            new_name = f"{prefix}.{new_number}{file_extension}"

            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            try:
                os.rename(old_path, new_path)
                print(f"Renomeado: {filename} -> {new_name}")
            except Exception as e:
                print(f"Erro ao renomear {filename}: {e}")



# Configurar o diretório e número inicial
input_directory = "/Projetos/2024_Phd_ObstacleDetectionModel/via_clear/correct"
start_number = 175  # Altere para o número inicial desejado
clear_nonclear = "clear"
# Executar a renomeação
#rename_images(input_directory, start_number, clear_nonclear)

input_directory = "/Projetos/2024_Phd_ObstacleDetectionModel/via_clear"
# Adicionado zero a esquerda no nome do arquivo
add_leading_zero(input_directory)
