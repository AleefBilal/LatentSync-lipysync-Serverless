

# LatentSync LipSync – Despliegue Serverless y Local

Este repositorio proporciona un **despliegue listo para serverless y compatible con ejecución local** del modelo de sincronización labial **LatentSync 1.6 de ByteDance**.
Soporta la **selección explícita de entorno** para despliegues **locales**, de **staging** y de **producción**.
Este sistema fue ejecutado y probado en **Nvidia RTX 3090 y A40** y consumió **~19 GB de memoria de vídeo (VRAM)**
---

## 🚀 Características Principales

* Inferencia GPU serverless (compatible con RunPod)
* Selección explícita de **entorno** (`local`, `stag`, `prod`)
* Entorno CUDA containerizado con Docker
* Modelos precargados (UNet, Whisper, VAE, InsightFace)
* Sin descargas de modelos en tiempo de ejecución
* Reutilización global del pipeline
* Un algoritmo de aleatorización especializado para que cada próximo vídeo sea un vídeo nuevo.
* Limpieza ordenada en tiempo de ejecución y gestión de memoria GPU

---

## 🎬 Demo – Antes y Después (LatentSync)

### Antes (Vídeo de Entrada)

[https://github.com/user-attachments/assets/4a9bcf74-76a7-4109-9d52-ed91fb7b3239](https://github.com/user-attachments/assets/4a9bcf74-76a7-4109-9d52-ed91fb7b3239)

### Después (Salida de LatentSync)

[https://github.com/user-attachments/assets/dfdab143-d3b6-4da7-ab69-e343f18928e6](https://github.com/user-attachments/assets/dfdab143-d3b6-4da7-ab69-e343f18928e6)

---

## 🔧 Niveles de Entorno (Obligatorio)

> **Importante:**
> El campo `level` es **obligatorio** para todas las ejecuciones.

### 🖥️ Local (Desarrollo / Depuración)

```json
{
  "level": "local",
  "ref_video_path": "/absolute/path/to/video.mp4",
  "ref_audio_path": "/absolute/path/to/audio.wav"
}
```

* Utiliza el sistema de archivos local
* No requiere credenciales en la nube
* Destinado únicamente a desarrollo y depuración

---

### ☁️ Staging (AWS)

```json
{
  "level": "stag",
  "ref_video_path": "s3://staging-bucket/path/video.mp4",
  "ref_audio_path": "s3://staging-bucket/path/audio.wav"
}
```

* Utiliza recursos de AWS de staging
* Credenciales y buckets separados
* Refleja la configuración de producción de forma segura

---

### 🚀 Producción (AWS)

```json
{
  "level": "prod",
  "ref_video_path": "s3://production-bucket/path/video.mp4",
  "ref_audio_path": "s3://production-bucket/path/audio.wav"
}
```

* Utiliza la infraestructura de AWS de producción
* Políticas estrictas de acceso e IAM
* Destinado a cargas de trabajo en vivo

---

## 🧪 Modo de Información / Verificación de Estado (Health Check)

```json
{
  "aleef": true
}
```

Devuelve los metadatos del servicio sin ejecutar la inferencia.

---

## 📁 Estructura del Repositorio

```
.
├── app.py
├── Dockerfile
├── requirements.txt
├── utils/
├── LatentSync/
├── checkpoints/
└── test_input.json
```

---

## 📦 Construcción de Docker (Build)

```bash
docker build -t latentsync-lipsync-serverless .
```

Todos los modelos están **precargados en el momento de la compilación**, garantizando una ejecución en tiempo de ejecución totalmente sin conexión.

---

## 🛠 Tecnologías Utilizadas (Tech Stack)

* Python 3.10
* PyTorch (CUDA)
* Diffusers
* LatentSync 1.6
* Whisper
* InsightFace
* RunPod Serverless
* AWS S3

---

## 🧹 Comportamiento en Tiempo de Ejecución

* Archivos temporales creados en `/tmp`
* Memoria GPU liberada después de cada trabajo
* Pipeline global reutilizado entre invocaciones en caliente (warm)

---

## 📄 Licencia

* LatentSync: Apache 2.0
* Otras dependencias siguen las licencias de sus proyectos originales

---

## ✅ Estado

✔ Soporte para modos local, staging y producción
✔ Imagen Docker serverless desplegada
✔ Modelos precargados y fijados

---
🙏 Agradecimientos

Un agradecimiento especial y sincero al equipo de ByteDance LatentSync por su destacado trabajo en este modelo.
Este despliegue se basa en su investigación y excelencia en ingeniería, y reconocemos su contribución con profundo respeto y gratitud.

### Ejecutar en local
```bash
sudo docker run --rm -it   --runtime=nvidia   --gpus all   -e NVIDIA_VISIBLE_DEVICES=all   -e NVIDIA_DRIVER_CAPABILITIES=video,compute,utility   lat_t_1
```
