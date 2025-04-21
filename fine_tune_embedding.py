# fine_tune_embedding.py
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

# Set environment variables 
os.environ['PYTHONHASHSEED'] = '42'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = random_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create a generator for DataLoader shuffling reproducibility
g = torch.Generator()
g.manual_seed(random_seed)

# Initialize the embedding model
embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

# Enhanced training data with more specific airline-related examples
train_examples = improved_train_examples = improved_train_examples = [
    InputExample(texts=["The flight was delayed by 3 hours", "My flight experienced a significant delay"]),
    InputExample(texts=["Excellent cabin service", "Outstanding flight attendants"]),
    InputExample(texts=["Premium economy seats were uncomfortable", "Upgraded seating was disappointing"]),
    InputExample(texts=["Food was terrible and cold", "Awful meal quality and presentation"]),
    InputExample(texts=["Cabin crew was helpful during medical emergency", "Flight attendants assisted with my special needs"]),
    InputExample(texts=["COVID-19 safety measures were inadequate", "Pandemic protocols were strictly enforced"]),
    InputExample(texts=["Refund policy was unclear and frustrating", "Ticket cancellation process was complicated"]),
    InputExample(texts=["Business class amenities were outdated", "Luxury travel experience fell short of expectations"]),
    InputExample(texts=["In-flight entertainment system was excellent", "Movie selection was disappointing"]),
    InputExample(texts=["Baggage was damaged during handling", "Lost luggage experience"]),
    InputExample(texts=["Check-in process was efficient", "Boarding procedures were chaotic"]),
    InputExample(texts=["Value for money in economy class", "Overpriced tickets for the service provided"]),
    InputExample(texts=["Aircraft cleanliness standards", "Bathroom maintenance during long-haul flight"]),
    InputExample(texts=["Seat comfort on overnight flights", "Sleeping quality in different cabin classes"]),
    InputExample(texts=["Flight cancellation communication", "Schedule change notification issues"]),
    
    # Aircraft-specific examples
    InputExample(texts=["A380 upper deck experience", "A380 double-decker aircraft comfort"]),
    InputExample(texts=["Boeing 777 cabin layout", "777-300ER flight experience"]),
    InputExample(texts=["A350 cabin pressure and humidity", "A350 modern amenities"]),
    InputExample(texts=["787 Dreamliner window features", "787 reduced jet lag experience"]),
    
    # Airport-specific examples
    InputExample(texts=["Changi transit experience", "Singapore airport facilities"]),
    InputExample(texts=["Heathrow Terminal check-in experience", "London airport security process"]),
    InputExample(texts=["Sydney Airport lounge access", "Australian airport immigration efficiency"]),
    InputExample(texts=["JFK Terminal boarding process", "New York departure experience"]),
    InputExample(texts=["Frankfurt Airport transit confusion", "German airport walking distances"]),
    
    # Cabin class-specific examples
    InputExample(texts=["First class suite privacy", "First class exclusive service"]),
    InputExample(texts=["Business class lie-flat bed comfort", "Business class lounge access"]),
    InputExample(texts=["Premium economy seat width and pitch", "Premium economy value proposition"]),
    InputExample(texts=["Economy class legroom constraints", "Economy class seat comfort"]),
    
    # Food and beverage specific examples
    InputExample(texts=["Book the Cook pre-order service", "Special meal requests"]),
    InputExample(texts=["Wine selection in premium cabins", "Champagne quality in first class"]),
    InputExample(texts=["Economy meal presentation", "Meal quantity and quality"]),
    InputExample(texts=["Breakfast service timing", "Mid-flight snack options"]),
    InputExample(texts=["Satay service in premium cabins", "Signature dishes quality"]),
    
    # Customer service specific examples
    InputExample(texts=["Refund processing timeframe", "Voucher compensation issues"]),
    InputExample(texts=["KrisFlyer miles crediting problems", "Loyalty program benefits"]),
    InputExample(texts=["Complaint handling process", "Customer service response time"]),
    InputExample(texts=["Flight change flexibility", "Rebooking assistance quality"]),
    
    # Accessibility examples
    InputExample(texts=["Wheelchair assistance at airports", "Special needs accommodation"]),
    InputExample(texts=["Accessibility for hearing impaired", "Mobility assistance during boarding"]),
    
    # Digital experience examples
    InputExample(texts=["Website booking interface", "Mobile app functionality"]),
    InputExample(texts=["Online check-in system", "Digital boarding pass issues"]),
    
    # COVID-specific examples
    InputExample(texts=["Pandemic safety protocols", "COVID-19 testing requirements"]),
    InputExample(texts=["Mask enforcement policies", "Social distancing measures"]),
    InputExample(texts=["Pandemic-related cancellation policy", "COVID flight schedule disruptions"]),
    
    # Crew service examples
    InputExample(texts=["Singapore Girl service standards", "Cabin crew attentiveness"]),
    InputExample(texts=["Flight attendant response to requests", "Crew professionalism"]),
    InputExample(texts=["Purser special assistance", "Crew language proficiency"]),
    
    # Comfort-specific examples
    InputExample(texts=["Seat padding quality", "Seat recline functionality"]),
    InputExample(texts=["Legroom in different aircraft", "Seat pitch comparison"]),
    InputExample(texts=["Noise levels during flight", "Sleep quality factors"]),
    InputExample(texts=["Cabin temperature control", "Air quality in cabin"]),
    
    # Entertainment examples
    InputExample(texts=["Movie selection variety", "TV show options"]),
    InputExample(texts=["Headphone quality", "Screen resolution and size"]),
    InputExample(texts=["In-flight WiFi performance", "Entertainment system responsiveness"]),
    
    # Value proposition examples
    InputExample(texts=["Price to service ratio", "Fare comparison with competitors"]),
    InputExample(texts=["Extra fees and charges", "Hidden cost complaints"]),
    InputExample(texts=["Upgrade value assessment", "Premium service worth"]),
]

# Create dataset and dataloader for training
train_dataset = SentencesDataset(train_examples, model=embedding_model)
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=16, 
    worker_init_fn=seed_worker,
    generator=g
)
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

# Fine-tune the embedding model
num_epochs = 50  
embedding_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=100,
    show_progress_bar=True
)

# Saving the fine-tuned embedding model for later reuse
embedding_save_path = "trained_embedding_model"
embedding_model.save(embedding_save_path)
print(f"Embedding model saved to {embedding_save_path}")
