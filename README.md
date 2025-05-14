# RAG Insurance Chatbot

Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) utilizando un dataset de seguros para crear un chatbot inteligente.

## Características

- Procesamiento por lotes de documentos
- Embeddings usando HuggingFace
- Almacenamiento vectorial con ChromaDB
- Motor de consultas con LLM

## Configuración

1. Crea un entorno virtual:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura las variables de entorno:
- Crea un archivo `.env` en la raíz del proyecto
- Añade tu token de HuggingFace:
```
HUGGINGFACE_TOKEN=tu_token_aqui
```

## Uso

1. Ejecuta el script principal:
```bash
python main.py
```

2. El script realizará las siguientes operaciones:
- Carga del dataset de seguros
- Procesamiento de documentos
- Creación de embeddings
- Configuración de la base de datos vectorial
- Preparación del motor de consultas

## Estructura del Proyecto

- `main.py`: Script principal
- `config.py`: Configuración centralizada
- `requirements.txt`: Dependencias del proyecto
- `.env`: Variables de entorno (no incluido en el repositorio)
- `data/`: Directorio para archivos de texto procesados
- `process.log`: Registro de operaciones
