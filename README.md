# opencv-project
## Datasets
Folder que contiene los datasets, luego de correr "main.py" se generan más datasets con las imágenes procesadas para cada uno de los tipos de procesamiento que se quiere utilizar en el modelo.
## EDA
Folder que contiene el script de python utilizado para realizar EDA sobre las imágenes.
## image_processing
Folder que contiene "processing.py", este script de python contiene los métodos con nuestra implementación de las funciones de opencv. los métodos implementados reciben el path de una imágen, la modifican utilizando la interfaz de OpenCV, y la retornan.
## model
Folder que contiene el script de python que genera y entrena el modelo para realizar la evaluación de ellos con los disintos tipos de procesamiento de imágenes, además de extraer las métricas del modelo.

## main.py

main del proyecto, llamando se generan los datasets con las imágenes procesadas, así como entrenar y guardar distintos los distintos modelos entrenados con ellos.