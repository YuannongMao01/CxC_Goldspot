from tqdm import tqdm
from util import *
from model import *
from torch.utils.data.dataloader import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EPOCH:12 VERSION:2.0——Accuracy is 36.8%
# EPOCH:16 VERSION:2.0—— Accuracy is 18%
# EPOCH:8 VERSION:3.0——Accuracy is 19%    
EPOCH = 12
VERSION = 2.0

dataset = IamgesWithLabelsDataset('/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/')
train_dataset, test_dataset = split_train_test(dataset)
loader_dataset = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0)

model = C3D(3)

# Loading the model
PATH = f'/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/{VERSION}-{EPOCH}-c3d.pt'
m_state_dict = torch.load(PATH)
model.load_state_dict(m_state_dict)

pbar = tqdm(loader_dataset, total=len(loader_dataset))

T = 0
A = 0
for cnt, batch in enumerate(pbar):
    frame, label = batch
    frame = frame.permute(0, 2, 1, 3, 4)
    prd = F.softmax(model(frame))
    target = onehot2label(label[0])
    source = onehot2label(torch.argmax(prd[0]))
    if (target == source): T += 1
    A += 1
    pbar.set_description(f'True:{target},' ',Predict:{source},' ',Accuracy:{T*100/A}%')

