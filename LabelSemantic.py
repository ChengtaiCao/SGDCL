"""
    Implementation of Get Embedding from Pretrained BERT for Label Semantic Information 
"""

import torch
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# action labels & reason labels
ACTIONS = [
    "forward", "stop or slow down", "left", "right"
]
REASONS = [
    "follow traffic", "the road is clear", "the traffic light is green",                                                        # 3
    "obstacle: car", "obstacle: person or pedestrain", "obstacle: rider", "obstacle: others", "traffic light", "traffic sign",  # 6
    "front car turning left", "on the left-turn lane", "traffic light allows left",                                             # 3
    "front car turning right", "on the right-turn lane", "traffic light allows right",                                          # 3
    "obstacles on the left lane", "no lane on the left", "solid line on the left",                                              # 3
    "obstacles on the right lane", "no lane on the right", "solid line on the right"                                            # 3
]


if __name__ == "__main__":
    # get label embeddings
    model = SentenceTransformer('all-mpnet-base-v2')

    label_embeddings = []

    for item in tqdm(ACTIONS + REASONS):
        sentence = item
        embedding = model.encode(sentence)
        embedding = torch.tensor(embedding, dtype=torch.float32)
        label_embeddings.append(embedding)
    
    # stack to a tensor
    label_embeddings = torch.stack(label_embeddings, dim=0)

    # store embedding
    file_name = "./label_embedding.pkl"
    with open(f"{file_name}", "wb") as f:
        pickle.dump(label_embeddings, f)
    print(f"Finish get label embedding and storing it in {file_name}")
