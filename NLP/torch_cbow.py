import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
Practice for generating CBOW embeddings from pytorch website
'''
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []

for i in range(2, len(raw_text) - 2):
    context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
    data.append((context, raw_text[i]))

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size):
        super(CBOW, self).__init__()
        self.Embeddings = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(context_size*embed_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.Embeddings(inputs).view(1, -1)
        print(embeds.shape)
        out = self.linear1(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = CBOW(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE*2)
optimizer = optim.SGD(model.parameters(), lr=1e-3)

for e in range(1):
    total = 0
    for context, target in data:
        context_ids = torch.tensor([word_to_ix[w] for w in context],\
                dtype=torch.long)
        model.zero_grad()
        log_prob = model(context_ids)
        loss = loss_function(log_prob, torch.tensor([word_to_ix[target]],\
                dtype=torch.long))
        loss.backward()
        optimizer.step()
        total += loss.item()
    losses.append(total)
print(losses)

