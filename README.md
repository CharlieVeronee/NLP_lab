# Natural Language Processing Models in PyTorch

- project explores:
  - use of pre-trained word embeddings (ConceptNet Numberbatch) and various neural models for NLP tasks
  - AG News dataset and SWEM with Embedding model for text classification

# 1. Exploration and analogy solving using embeddings

## Dataset

- ConceptNet Numberbatch 2016 (mini.h5)
- Contains 362,891 word vectors (300-dimensional)

## Steps

- Download & extract English words
- Normalize vectors
- Build word index for fast lookup

## Functions

- similarity_score(w1, w2) — Computes cosine similarity between two words
- most_similar(w, n) — Finds n most similar words to a given word
- solve_analogy(a1, b1, a2) — Solves analogies like "man : king :: woman : ?"

# 2. Simple Word Embedding Model (SWEM)

- for sentiment analysis to classify movie reviews as positive or negative

## Preprocessing

- Remove punctuation and lowercase text
- Map known words to vectors
- Use mean pooling across word embeddings

## Model

Class SWEM(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(300, 64)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))

        return self.fc2(x)

- Loss: nn.BCEWithLogitsLoss()
- Optimizer: torch.optim.Adam

## Accuracy

- ~97.9% test accuracy after 250 epochs

# 3. SWEM With Learned Embeddings

- Uses nn.Embedding to learn embeddings from scratch
- Input is now word indices, not pre-computed vectors

# 4. AG News + Text Classification

## Load Data

- Inputs: Tokenized and padded sequences of word indices
- Labels: Integer class labels (0-3)
- A custom collate_fn is used to prepare batches for trainin

## Model

Class SWEM(nn.Module):

    def __init__(self, vocab_size, embedding_dimensions, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimensions)
        self.fc1 = nn.Linear(embedding_dimensions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):

        embed = self.embedding(x)
        embed_mean = torch.mean(embed, dim=1)
        h = self.fc1(embed_mean)
        h = F.relu(h)
        h = self.fc2(h)
        return h

- Loss: CrossEntropyLoss
- Optimizer: torch.optim.Adam

## Accuracy

- ~90.7% test accuracy after 3 epochs
