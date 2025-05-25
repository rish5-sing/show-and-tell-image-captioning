import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
import torchvision.models as models
import math
import torch.utils.data as data

nltk.download('punkt')

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return word_tokenize(text.lower()) 

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
        

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def get_train_indices(self, split_ratio=0.8):
        total_samples = len(self)
        train_size = int(split_ratio * total_samples)
        indices = list(range(total_samples))
        return indices[:train_size]

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    
    
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


batch_size = 128 
vocab_threshold = 5 
embd_size = 256 
hidden_size = 512
num_epochs = 10
print_every = 10
save_every = 1
log_file = 'training_log.txt'

def get_loader(
    root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


class CNN(nn.Module):
    def __init__(self , embd_size):
        super(CNN , self).__init__()
        resnet = models.resnet50(pretrained = True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embd = nn.Linear(resnet.fc.in_features, embd_size)

    def forward(self , images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embd(features)
        return features
    

class RNN(nn.Module):
    def __init__(self, embd_size , hidden_size , vocab_size , num_layers = 1):
        super(RNN , self).__init__()

        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embd_size)
        self.lstm = nn.LSTM(embd_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self , features , captions):
        cap_embedding = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(dim=0), cap_embedding), dim=0)
        lstm_out, self.hidden = self.lstm(embeddings)  
        outputs = self.linear(lstm_out)  

        return outputs
    
loader , dataset = get_loader('./Images/', './captions.txt', transform)

class CNNtoRNN(nn.Module):
    def __init__(self, embd_size, hidden_size, vocab_size, num_layers= 1):
        super(CNNtoRNN, self).__init__()
        self.CNN = CNN(embd_size)
        self.RNN = RNN(embd_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.CNN(images)
        outputs = self.RNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.CNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.RNN.lstm(x, states)
                output = self.RNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.RNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
    

vocab_size = len(dataset.vocab)

model = CNNtoRNN(embd_size, hidden_size, vocab_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = (nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]).cuda() if torch.cuda.is_available 
             else nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]))

params = list(model.parameters())

optimizer = torch.optim.Adam(params, lr = 0.01)

total_step = math.ceil(len(dataset) / loader.batch_sampler.batch_size)


f = open(log_file, 'w')

for epoch in range(1, num_epochs+ 1):
    for i in range(1 , total_step+1):
        indices = loader.dataset.get_train_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices = indices)
        loader.batch_sampler.sampler = new_sampler

        images , captions = next(iter(loader))
        images = images.to(device)
        captions = captions.to(device)

        model.zero_grad()

        outputs = model(images , captions[:-1])
        
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        loss.backward()
        optimizer.step()

        stats = (
            f"Epoch [{epoch}/{num_epochs}], Step [{i}/{total_step}],"
            f"Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
        )

        f.write(stats + '\n')
        f.flush()

        if i % print_every == 0:
            print("\r" + stats)

    if epoch % save_every == 0:
        torch.save(model.state_dict(), os.path.join('./models', f'model-{epoch}.pkl'))

f.close()


model_file = 'model-10.pkl'

model.load_state_dict(torch.load(os.path.join('./models', model_file)))

model.eval()

transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

test_image = transform(Image.open("./Images/3726629271_7639634703.jpg").convert("RGB")).unsqueeze(
        0
    )
with torch.no_grad():
    output = model.caption_image(test_image.to(device), dataset.vocab)
    
print(output)
