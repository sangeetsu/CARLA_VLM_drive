import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Model
import numpy as np
import os
from PIL import Image
import pickle
import time
from typing import List, Dict, Union, Tuple, Optional

class BLIPEmbedder:
    """
    Class for generating BLIP2 embeddings for RGB frames captured at waypoints.
    This implementation follows the integration strategy outlined in the 
    'Integrating CLIP Embeddings for Human-Aligned Controller Optimization' document.
    """
    
    def __init__(self, model_name: str = "Salesforce/blip2-flan-t5-xl", device: str = None):
        """
        Initialize the BLIP2 embedding generator.
        
        Args:
            model_name: The BLIP2 model to use
            device: Computing device (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing BLIP2 model on {self.device}")
        
        # Load model and processor
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2Model.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Cache for embedding storage
        self.embedding_cache = {}
        
        # Set model to evaluation mode
        self.model.eval()
        
    def generate_embeddings(self, 
                           frames: List[np.ndarray], 
                           frame_timestamps: List[float] = None, 
                           batch_size: int = 4) -> Dict[float, np.ndarray]:
        """
        Generate BLIP2 embeddings for a batch of frames.
        
        Args:
            frames: List of RGB frames as numpy arrays
            frame_timestamps: List of timestamps corresponding to frames
            batch_size: Number of frames to process in one batch
        
        Returns:
            Dictionary mapping timestamps to embeddings
        """
        if frame_timestamps is None:
            frame_timestamps = list(range(len(frames)))
            
        assert len(frames) == len(frame_timestamps), "Number of frames must match number of timestamps"
        
        timestamp_to_embedding = {}
        
        # Process batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_timestamps = frame_timestamps[i:i+batch_size]
            
            # Convert numpy arrays to PIL Images
            pil_images = [Image.fromarray(frame) for frame in batch_frames]
            
            # Generate embeddings
            with torch.no_grad():
                # Process images with the processor which handles both image and text inputs
                inputs = self.processor(images=pil_images, text=[""] * len(pil_images), return_tensors="pt", padding=True).to(self.device)
                
                # Get the vision encoder outputs (this is different than just passing to model)
                # We use vision_outputs directly instead of the full model
                vision_outputs = self.model.vision_model(
                    pixel_values=inputs.pixel_values,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                
                # Get image embeddings from the vision outputs
                image_embeds = vision_outputs.pooler_output
                
                # Normalize embeddings
                image_embeds = image_embeds / torch.norm(image_embeds, dim=1, keepdim=True)
                
                # Convert to numpy and store with timestamps
                embeddings = image_embeds.cpu().numpy()
                
                for j in range(len(batch_timestamps)):
                    timestamp_to_embedding[batch_timestamps[j]] = embeddings[j]
            
        return timestamp_to_embedding
    
    def align_embeddings_with_telemetry(self, 
                                       embeddings_dict: Dict[float, np.ndarray], 
                                       telemetry_data: Dict[float, Dict]) -> Dict[float, Dict]:
        """
        Align embeddings with vehicle telemetry data using timestamps.
        
        Args:
            embeddings_dict: Dictionary mapping timestamps to embeddings
            telemetry_data: Dictionary mapping timestamps to telemetry measurements
            
        Returns:
            Dictionary containing aligned embeddings and telemetry data
        """
        aligned_data = {}
        
        for timestamp in embeddings_dict:
            # Find closest telemetry timestamp
            closest_timestamp = min(telemetry_data.keys(), key=lambda x: abs(x - timestamp))
            
            # Combine data
            aligned_data[timestamp] = {
                "embedding": embeddings_dict[timestamp],
                "telemetry": telemetry_data[closest_timestamp],
                "telemetry_timestamp": closest_timestamp
            }
            
        return aligned_data
    
    def save_embeddings(self, embeddings_dict: Dict, filepath: str) -> None:
        """
        Save embeddings to a file for later use.
        
        Args:
            embeddings_dict: Dictionary containing embeddings and associated data
            filepath: Path to save the embeddings
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict:
        """
        Load previously saved embeddings.
        
        Args:
            filepath: Path to the saved embeddings file
            
        Returns:
            Dictionary containing the loaded embeddings
        """
        try:
            with open(filepath, 'rb') as f:
                embeddings_dict = pickle.load(f)
            print(f"Embeddings loaded from {filepath}")
            return embeddings_dict
        except OSError as e:
            if e.errno == 24:  # Too many open files
                print(f"Too many open files error. Trying to garbage collect and retry...")
                import gc
                gc.collect()  # Force garbage collection
                # Try again after cleanup
                with open(filepath, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                print(f"Successfully loaded embeddings after garbage collection")
                return embeddings_dict
            else:
                # Re-raise if it's a different error
                raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def align_trajectories(self, 
                           actual_embeddings: List[np.ndarray], 
                           simulated_embeddings: List[np.ndarray]) -> float:
        """
        Implement Dynamic Time Warping (DTW) for embedding alignment as described in the document.
        
        Args:
            actual_embeddings: List of embeddings from actual driving
            simulated_embeddings: List of embeddings from simulated driving
            
        Returns:
            DTW similarity score
        """
        n = len(actual_embeddings)
        m = len(simulated_embeddings)
        
        # Initialize DTW matrix
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, 1:] = float('inf')
        dtw_matrix[1:, 0] = float('inf')
        
        # Fill the DTW matrix
        for i in range(1, n+1):
            for j in range(1, m+1):
                # Cost is 1 - cosine similarity
                cost = 1 - self.compute_similarity(actual_embeddings[i-1], simulated_embeddings[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
        
        return dtw_matrix[n, m]

    def get_fitness_component(self, 
                            actual_embeddings: List[np.ndarray], 
                            simulated_embeddings: List[np.ndarray]) -> float:
        """
        Calculate the fitness component based on BLIP2 embedding similarity.
        This can be used with the GA_PID fitness function.
        
        Args:
            actual_embeddings: List of embeddings from actual driving
            simulated_embeddings: List of embeddings from simulated driving
            
        Returns:
            Similarity score (higher is better)
        """
        dtw_score = self.align_trajectories(actual_embeddings, simulated_embeddings)
        # Convert DTW score to similarity score (lower DTW means higher similarity)
        similarity_score = 1 / (1 + dtw_score)
        return similarity_score

# Example usage in GA_PID context
if __name__ == "__main__":
    # Initialize embedder
    embedder = BLIPEmbedder()
    
    # Example: Generate embeddings from a list of frames
    # frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
    # timestamps = [time.time() + i for i in range(len(frames))]
    # embeddings = embedder.generate_embeddings(frames, timestamps)
    
    # Example: Save embeddings
    # embedder.save_embeddings(embeddings, "embeddings/test_embeddings.pkl")
    
    print("BLIP2 embedder initialized successfully. Ready for integration with GA_PID.")