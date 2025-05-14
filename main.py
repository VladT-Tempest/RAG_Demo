from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import nest_asyncio
import asyncio
import logging
import chromadb
from typing import List, Optional
from datasets import load_dataset
from config import *
import aiohttp
from contextlib import AsyncExitStack

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HttpSessionManager:
    _instance: Optional['HttpSessionManager'] = None
    _session: Optional[aiohttp.ClientSession] = None
    _exit_stack: Optional[AsyncExitStack] = None

    def __init__(self):
        if HttpSessionManager._instance is not None:
            raise RuntimeError("Use HttpSessionManager.instance() instead")
        self._exit_stack = AsyncExitStack()

    @classmethod
    def instance(cls) -> 'HttpSessionManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            await self._exit_stack.enter_async_context(self._session)
        return self._session

    async def cleanup(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._session = None
            HttpSessionManager._instance = None

async def get_http_session() -> aiohttp.ClientSession:
    """Obtiene una sesión HTTP compartida"""
    return await HttpSessionManager.instance().get_session()

def chunks(lst, n):
    """Divide una lista en lotes de tamaño n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def process_batch(pipeline, documents: List, batch_num: int) -> List:
    """Procesa un lote de documentos con reintentos"""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Procesando lote {batch_num} (intento {attempt + 1})")
            nodes = await pipeline.arun(documents=documents)
            logger.info(f"Lote {batch_num} procesado exitosamente")
            return nodes
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Error en lote {batch_num}, reintentando en {RETRY_DELAY} segundos: {str(e)}")
                await asyncio.sleep(RETRY_DELAY)
            else:
                logger.error(f"Error fatal en lote {batch_num} después de {MAX_RETRIES} intentos: {str(e)}")
                raise

async def load_and_prepare_dataset():
    """Carga el dataset y prepara los archivos de texto"""
    dataset = load_dataset(DATASET_NAME, split="train").select(range(MAX_RECORDS))
    logger.info(f"Dataset cargado con {len(dataset)} registros")

    Path("data").mkdir(parents=True, exist_ok=True)
    logger.info("Directorio 'data' creado o verificado")

    for i, data in enumerate(dataset):
        with open(Path("data") / f"data_{i}.txt", "w", encoding='utf-8') as f:
            text_to_write = " ".join([data['instruction'], data['intent'], data['category'], data['tags'], data['response']])
            f.write(text_to_write)
    logger.info(f"Se escribieron {len(dataset)} archivos de texto")
    return dataset

async def create_and_process_pipeline(documents):
    """Crea y ejecuta el pipeline de procesamiento"""
    session = await get_http_session()
    embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name=EMBEDDING_MODEL, 
        token=HUGGINGFACE_TOKEN,
        session=session
    )
    
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embed_model,
        ]
    )

    all_nodes = []
    chunk_size = min(BATCH_SIZE, len(documents))
    
    for batch_num, batch in enumerate(chunks(documents, chunk_size)):
        nodes = await process_batch(pipeline, batch, batch_num)
        all_nodes.extend(nodes)
        if batch_num < len(documents) // chunk_size - 1:
            await asyncio.sleep(1)
    
    return all_nodes

async def setup_vector_store(documents):
    """Configura y prepara el vector store"""
    try:
        logger.info("Iniciando configuración del Vector Store...")
        session = await get_http_session()
        
        # Configurar ChromaDB
        logger.info(f"Configurando ChromaDB en {CHROMA_DB_PATH}")
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Limpiar colección existente si existe
        try:
            db.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Colección anterior '{COLLECTION_NAME}' eliminada")
        except Exception as e:
            logger.info(f"No se encontró colección anterior: {str(e)}")
        
        # Crear nueva colección con configuración específica
        chroma_collection = db.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Insurance chatbot data"}
        )
        logger.info(f"Nueva colección '{COLLECTION_NAME}' creada")
        
        # Crear vector store
        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )
        logger.info("Vector Store creado")
        
        # Configurar modelo de embeddings
        logger.info("Configurando modelo de embeddings...")
        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name=EMBEDDING_MODEL,
            token=HUGGINGFACE_TOKEN,
            session=session
        )

        # Configurar pipeline optimizado
        logger.info("Configurando pipeline de ingestión...")
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    paragraph_separator="\n\n"
                ),
                embed_model,
            ],
            vector_store=vector_store
        )

        # Procesar documentos en lotes más pequeños
        logger.info(f"Procesando {len(documents)} documentos...")
        total_documents = len(documents)
        batch_size = min(100, total_documents)
        
        for i in range(0, total_documents, batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"Procesando lote {i//batch_size + 1} de {(total_documents + batch_size - 1)//batch_size}")
            await pipeline.arun(documents=batch)
            
            # Verificar progreso
            current_count = chroma_collection.count()
            logger.info(f"Documentos procesados hasta ahora: {current_count}")
        
        # Verificación final
        final_count = chroma_collection.count()
        logger.info(f"Total de documentos procesados y guardados en ChromaDB: {final_count}")
        
        if final_count == 0:
            raise Exception("No se guardaron documentos en ChromaDB")
            
        return vector_store, embed_model

    except Exception as e:
        logger.error(f"Error en setup_vector_store: {str(e)}", exc_info=True)
        raise

async def setup_query_engine(vector_store, embed_model):
    """Configura el motor de consultas"""
    session = await get_http_session()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    nest_asyncio.apply()
    llm = HuggingFaceInferenceAPI(
        model_name=LLM_MODEL, 
        token=HUGGINGFACE_TOKEN,
        session=session
    )
    return index.as_query_engine(
        response_mode="tree_summarize",
        llm=llm,
    )

async def process_documents():
    """Función principal de procesamiento de documentos"""
    try:
        session = await get_http_session()
        await load_and_prepare_dataset()
        
        reader = SimpleDirectoryReader("data")
        documents = reader.load_data()
        logger.info(f"Documentos cargados: {len(documents)}")
        
        logger.info("Configurando vector store y procesando documentos...")
        vector_store, embed_model = await setup_vector_store(documents)
        
        logger.info("Configurando motor de consultas")
        query_engine = await setup_query_engine(vector_store, embed_model)
        
        logger.info(f"Pipeline completado exitosamente")
        return len(documents), query_engine
            
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {str(e)}", exc_info=True)
        raise
    finally:
        await HttpSessionManager.instance().cleanup()

async def main():
    try:
        num_nodes, query_engine = await process_documents()
        logger.info(f"Procesamiento completado. Número de nodos procesados: {num_nodes}")
        
        # Ejemplo de consulta
        response = query_engine.query(
            "How can i get an auto insurance?"
        )
        print(response)

    except Exception as e:
        logger.error("Error en la ejecución principal", exc_info=True)
        raise
    finally:
        await HttpSessionManager.instance().cleanup()

if __name__ == "__main__":
    asyncio.run(main())