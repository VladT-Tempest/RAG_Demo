from pathlib import Path
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import nest_asyncio
import asyncio
import logging
import chromadb
from config import *
import aiohttp
from contextlib import AsyncExitStack

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HttpSessionManager:
    _instance = None
    _session = None
    _exit_stack = None

    def __init__(self):
        if HttpSessionManager._instance is not None:
            raise RuntimeError("Use HttpSessionManager.instance() instead")
        self._exit_stack = AsyncExitStack()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
            await self._exit_stack.enter_async_context(self._session)
        return self._session

    async def cleanup(self):
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._session = None
            HttpSessionManager._instance = None

async def get_http_session():
    return await HttpSessionManager.instance().get_session()

async def setup_query_engine():
    """Configura el motor de consultas desde ChromaDB existente"""
    try:
        session = await get_http_session()
        
        # Cargar ChromaDB existente
        logger.info("Cargando ChromaDB...")
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Intentar obtener la colección existente
        try:
            chroma_collection = db.get_collection(name=COLLECTION_NAME)
            collection_count = chroma_collection.count()
            logger.info(f"Colección encontrada con {collection_count} documentos")
        except Exception as e:
            logger.error(f"Error al acceder a la colección: {str(e)}")
            raise Exception("No se encontró la colección en ChromaDB. Ejecuta primero main.py para crear la base de datos.")
        
        # Verificar que la colección tenga datos
        if collection_count == 0:
            raise Exception("La colección está vacía. Ejecuta main.py para procesar los documentos.")
        
        # Crear vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        logger.info("Vector Store creado exitosamente")
        
        # Configurar el modelo de embeddings
        logger.info("Configurando modelo de embeddings...")
        embed_model = HuggingFaceInferenceAPIEmbedding(
            model_name=EMBEDDING_MODEL,
            token=HUGGINGFACE_TOKEN,
            session=session
        )

        # Crear índice desde el vector store
        logger.info("Creando índice...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # Configurar LLM con parámetros optimizados
        logger.info("Configurando LLM...")
        nest_asyncio.apply()
        llm = HuggingFaceInferenceAPI(
            model_name=LLM_MODEL,
            token=HUGGINGFACE_TOKEN,
            session=session,
            temperature=0.7,
            max_new_tokens=512,
            top_p=0.9,
            context_window=3072,
            api_kwargs={"wait_for_model": True}  # Asegurar que el modelo esté listo
        )
        
        # Configurar y retornar el query engine
        logger.info("Configurando query engine...")
        query_engine = index.as_query_engine(
            response_mode="compact",
            llm=llm,
            similarity_top_k=5,
            streaming=False
        )
        
        logger.info("Motor de consultas configurado exitosamente")
        return query_engine

    except Exception as e:
        logger.error(f"Error durante la configuración: {str(e)}", exc_info=True)
        raise

async def main():
    try:
        print("Iniciando sistema de consultas...")
        query_engine = await setup_query_engine()
        print("Sistema listo para consultas.")
        
        while True:
            print("\nEscribe tu pregunta (o 'salir' para terminar):")
            question = input().strip()
            
            if question.lower() == 'salir':
                break
            
            if not question:
                print("Por favor, escribe una pregunta válida.")
                continue
            
            print("\nBuscando respuesta...")
            try:
                response = query_engine.query(question)
                if not response or str(response).strip() == "":
                    print("No se encontró una respuesta relevante para tu pregunta.")
                else:
                    print("\nRespuesta:", response)
                    
            except Exception as e:
                print(f"\nError al procesar la consulta: {str(e)}")
                logger.error(f"Error en consulta: {str(e)}", exc_info=True)

    except Exception as e:
        logger.error("Error en la ejecución principal", exc_info=True)
        print(f"\nError: {str(e)}")
        raise
    finally:
        print("\nCerrando sesión...")
        await HttpSessionManager.instance().cleanup()
        print("Sesión cerrada.")

if __name__ == "__main__":
    asyncio.run(main())
