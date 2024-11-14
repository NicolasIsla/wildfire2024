import os

# Cargar la lista de imágenes a borrar
images_to_delete_file = '/home/nisla/wilfire2024/images_to_delete2.txt'
with open(images_to_delete_file, 'r') as f:
    images_to_delete = set(f.read().splitlines())

# Recorrer los datasets y eliminar imágenes y sus etiquetas
datasets_directories = [
    # '/data/nisla/Nemo/DS/images/train',
    # '/data/nisla/Nemo/DS/images/test',
    # '/data/nisla/Nemo/DS/images/val',
    "/data/nisla/TestSmokeFull/smoke_frame_test/images/test/"
]  # Lista de directorios de tus datasets

deleted_images_count = 0
deleted_labels_count = 0

for dataset_dir in datasets_directories:
    for image in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, image)

        name = os.path.basename(file_path)
            
        
        if name in images_to_delete:
            # Eliminar la imagen
            try:
                os.remove(file_path)
                deleted_images_count += 1
                print(f"Imagen eliminada: {file_path}")
            except Exception as e:
                print(f"Error al eliminar imagen {file_path}: {e}")

            # Buscar y eliminar la etiqueta correspondiente
            label_path = os.path.splitext(file_path)[0] + '.txt'
            # replace images with labels
            label_path = label_path.replace('images', 'labels')
            if os.path.exists(label_path):
                try:
                    os.remove(label_path)
                    deleted_labels_count += 1
                    print(f"Etiqueta eliminada: {label_path}")
                except Exception as e:
                    print(f"Error al eliminar etiqueta {label_path}: {e}")

print(f"Proceso completado. Total de imágenes eliminadas: {deleted_images_count}.")
print(f"Total de etiquetas eliminadas: {deleted_labels_count}.")
