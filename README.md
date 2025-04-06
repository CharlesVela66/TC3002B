# Weather Image Classification 🌦️  
Autor: **Carlos Velasco**

Este proyecto consiste en el desarrollo de un modelo de aprendizaje automático capaz de reconocer el tipo de clima presente en una imagen. Para ello se utiliza un conjunto de datos con imágenes etiquetadas por tipo de fenómeno climático, y se entrena una red neuronal simple (RNN) para realizar predicciones top-3 con probabilidades.

## 📁 Dataset
El dataset utilizado se encuentra disponible en Kaggle:  
🔗 [Weather Dataset - Kaggle](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)

Contiene 6862 imágenes divididas en 11 categorías de clima:

- dew  
- fog/smog  
- frost  
- glaze  
- hail  
- lightning  
- rain  
- rainbow  
- rime  
- sandstorm  
- snow  

## 🎯 Objetivo del Proyecto
Desarrollar un modelo que, dado una imagen de clima, sea capaz de predecir el tipo de fenómeno que aparece, devolviendo el top 3 de predicciones con su respectiva probabilidad.

## 🔧 Proceso del Proyecto

1. **Carga y visualización de datos**
2. **Selección de clases** (se puede trabajar solo con un subconjunto de categorías si se desea)
3. **Limpieza de datos** (revisión manual de calidad de imágenes)
4. **Transformación y normalización**
5. **División del dataset** en entrenamiento (70%), validación (15%) y prueba (15%)
6. **Entrenamiento de modelo** con una arquitectura básica de RNN
7. **Evaluación** usando accuracy y loss en cada epoch
8. **Mini app** para predecir imágenes personales cargadas por el usuario

## 🧠 Modelo
Se utilizó una red neuronal recurrente (RNN) simple, adaptada para procesar imágenes como secuencias. Este modelo permite capturar patrones espaciales básicos en las imágenes redimensionadas a 64x64 píxeles.

## 📂 Mini App
Al final del proyecto se incluye una sección que permite subir hasta 5 imágenes personales de clima y obtener las predicciones del modelo con sus probabilidades.

---

¡Gracias por revisar este proyecto! 🚀  
