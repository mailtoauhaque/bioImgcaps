import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from ptic_general_model import enc_dec
from get_loader import get_loader

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="/media/anwar/25ff85f1-4207-4d23-90c3-9ec96feba2cb/Flickr8K/Flickr8k_Images/",
        annotation_file="captions.txt",
        transform=transform,
        num_workers=8,

    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_size = 256
    hidden_size = 256
    vocab_size = dataset.__len__()
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 100

    writer = SummaryWriter("runs/flickr")
    step = 0

    model = enc_dec(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(num_epochs):
        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions[:-1])
            print(outputs)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            writer.add_scalar("Training Loss", loss.item(), global_step=step)

            step += 1
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()
        print("Epoch : ", epoch)


if __name__ == "__main__":
    train()
