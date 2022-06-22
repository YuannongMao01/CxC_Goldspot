from tqdm import tqdm
import numpy as np
from util import *
from model import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

# Enables benchmark mode in cudnn
# torch.backends.cudnn.benchmark = t = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_relearn = False
VERSION = 2.0   

dataset = IamgesWithLabelsDataset('/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/')   
train_dataset, test_dataset = split_train_test(dataset)


loader_dataset = DataLoader(train_dataset,
                            batch_size=10,   
                            shuffle=True,
                            num_workers=0)


model = C3D(3).to(device)

accumulate_steps = 2  
lr = 0.003
momentum = 0.9
EPOCH = 12
EPOCH_START = 0
average_loss = []
if is_relearn:
    EPOCH_START = 4
    PATH = f'/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/{VERSION}-{EPOCH_START}-c3d.pt'
    lr = 0.0003
    m_state_dict = torch.load(PATH)
    model.load_state_dict(m_state_dict)

optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      momentum=momentum,
                      weight_decay=5e-4)  

crossEntropyLoss = nn.CrossEntropyLoss()

losses = pd.DataFrame([],
                      columns=['Train Epoch', 'lr', 'loss', 'average_loss'])
lables = []

# training the model
for epoch in range(EPOCH_START, EPOCH):
    if (epoch + 1) % 4 == 0: lr /= 10
    pbar = tqdm(loader_dataset, total=len(loader_dataset))
    for cnt, batch in enumerate(pbar):
        frame, label = batch
        frame = frame.permute(0, 2, 1, 3, 4).to(device)
        prd = model(frame)
        # Expected floating point type for target with class probabilities, got Long
        loss = crossEntropyLoss(prd.to('cpu').float(), label)
        loss.backward()
        average_loss.append(loss.item())
        losses = losses.append(
            {
                'Train Epoch': f'{epoch + 1 } / {EPOCH}',
                'lr': lr,
                'loss': loss.item(),
                'average_loss': round(np.mean(average_loss), 4)
            },
            ignore_index=True)
        pbar.set_description(
            f'Train Epoch:{epoch + 1}/{EPOCH} lr:{lr} train_loss:{loss.item()} average_train_loss:{round(np.mean(average_loss), 4)}'
        )
        
        #To solve CUDA error: out of memory
        if (cnt + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        del loss
    if (epoch + 1) % 4 == 0:
        # The model are saved every four epochs
        PATH = f'/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/{VERSION}-{epoch + 1}-c3d.pt'
        torch.save(model.state_dict(), PATH)
        losses.to_csv(
            f'/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/{VERSION}-losses-lr-{lr} momentum-{momentum} epoch-{epoch}.csv'
        )
        del losses
        losses = pd.DataFrame(
            [], columns=['Train Epoch', 'lr', 'loss', 'average_loss'])
        del average_loss
        average_loss = []
