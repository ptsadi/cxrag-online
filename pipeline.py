from google.cloud import aiplatform
from google.oauth2 import service_account
import os
import pandas as pd
import PIL
from pinecone import Pinecone, ServerlessSpec
import io
import threading
import base64
import numpy as np

# Constants for Google Cloud API
ENDPOINT_ID = "6741963408964321280"
PROJECT_ID = "505839069664"
INPUT_DATA_FILE = "INPUT-JSON"
LOCATION = "asia-southeast2"

class CXRImageRetrieval:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, pinecone_api_key=None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CXRImageRetrieval, cls).__new__(cls)
                cls._instance._is_initialized = False
                cls._instance._pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        return cls._instance

    def __init__(self, pinecone_api_key=None):
        with self._lock:
            if self._is_initialized:
                return
            
            if not self._pinecone_api_key:
                raise ValueError("Pinecone API key is required")
            
            self._initialize()
            self._is_initialized = True

    def _initialize(self):
        """Initialize components"""
        self.index_name = "cxr-embeddings"
        
        # Initialize Pinecone
        pc = Pinecone(api_key=self._pinecone_api_key)
        try:
            self.index = pc.Index(self.index_name)
        except Exception:
            pc.create_index(
                name=self.index_name,
                dimension=4096,  # CXR Foundation embedding dimension
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                metric="cosine"
            )
            self.index = pc.Index(self.index_name)

        # Initialize Google Cloud AI Platform with credentials
        credentials = service_account.Credentials.from_service_account_file(
            'cxrag-446518-1841377c6ab2.json'
        )

        aiplatform.init(
            project=PROJECT_ID,
            location=LOCATION,
            credentials=credentials
        )
        
        # Set up the API endpoint for CXR Foundation
        self.endpoint = aiplatform.Endpoint(ENDPOINT_ID)

    def generate_embedding(self, image):
        """Generate embedding using Google's CXR Foundation API"""
        # Convert and compress image
        if isinstance(image, np.ndarray):
            pil_image = PIL.Image.fromarray(image)
        else:
            pil_image = image

        # Compress image while maintaining reasonable quality
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare request payload and make prediction
        instances = [{"input_bytes": img_str}]
        response = self.endpoint.predict(instances=instances)
        
        if not response.predictions:
            raise Exception("No predictions returned from the API")
        
        # Save the response locally
        with open('api_response.txt', 'a') as f:
            f.write(f"{response.predictions[0]}\n")
            
        contrastive_img_emb = response.predictions[0]["contrastive_img_emb"]
        # Convert list of lists to numpy array and flatten
        return np.array(contrastive_img_emb).flatten().tolist()

    def store_in_pinecone(self, embeddings_df: pd.DataFrame):
        """Store embeddings in Pinecone for faster retrieval."""
        try:
            # Try to delete existing vectors
            self.index.delete(deleteAll=True)
        except Exception:
            print(f"No existing vectors to delete")

        # Store embeddings in batches
        batch_size = 100
        try:
            for i in range(0, len(embeddings_df), batch_size):
                batch = embeddings_df.iloc[i:i + batch_size]
                vectors = [(
                    str(id),  # Pinecone requires string IDs
                    emb.tolist(),
                    {"type": "cxr_image"}
                ) for id, emb in zip(batch['image_id'], batch['embeddings'])]

                self.index.upsert(vectors=vectors)
            print(f"Successfully stored {len(embeddings_df)} embeddings in Pinecone")
        except Exception as e:
            raise Exception(f"Failed to store embeddings in Pinecone: {str(e)}")


    def convert_uncompressed_image_bytes_to_base64(image: np.ndarray) -> str:
        """Converts an uncompressed image array to a base64-encoded PNG string."""
        with io.BytesIO() as compressed_img_bytes:
            with PIL.Image.fromarray(image) as pil_image:
                pil_image.save(compressed_img_bytes, 'png')
        return base64.b64encode(compressed_img_bytes.getvalue())

