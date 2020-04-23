import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset_ssl import Dataset, MultiClassHingeLoss
from core.utils import set_device
from model.pennet import InpaintGenerator

LATEST_EPOCH = "/home/kanchana/Downloads/ref_repos/PEN-Net-for-Inpainting/release_model/pennet_pascal_64/gen_00022.pth"
BATCH_SIZE = 1
LR = 0.1
N_CLASS = 10
NUM_EPOCHS = 10
SAVE_PATH = "SSL_trained_models"
LOAD_MODEL = False


class Model(torch.nn.Module):
    """define network"""

    def __init__(self, n_feature=992, n_class=10):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(n_feature, n_class).cuda()
        torch.nn.init.kaiming_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.1)

        # build encoder network, load pre-trained weights, and freeze
        self.gen: InpaintGenerator = set_device(InpaintGenerator())
        data = torch.load(LATEST_EPOCH, map_location=lambda storage, loc: set_device(storage))
        self.gen.load_state_dict(data['netG'])
        for param in self.gen.parameters():
            param.requires_grad = False

    def forward(self, x, m):
        features, prediction = self.gen(x, m)
        average_pool = [torch.nn.functional.max_pool2d(feature, feature.shape[-1]).view(1, -1) for feature in features]
        combined = torch.cat(average_pool, dim=1)
        return self.fc(combined)


# load dataset
train_dataset = Dataset("", debug=False, split='train')  # num classes = 10 (0-9)
dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
model = Model(n_class=N_CLASS)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss = MultiClassHingeLoss()

training_f1 = 0
training_loss = 0

if LOAD_MODEL:
    svm_weights = torch.load("{}/svm_weights.pth".format(SAVE_PATH),
                             map_location=lambda storage, loc: set_device(storage))
    model.fc.load_state_dict(svm_weights["svm"])
    opt_weights = torch.load("{}/opt_weights.pth".format(SAVE_PATH),
                             map_location=lambda storage, loc: set_device(storage))
    optimizer.load_state_dict(opt_weights["opt"])

for epoch in range(NUM_EPOCHS):
    for idx, (im, mask, im_name, label) in tqdm(enumerate(dataloader)):
        im, mask = im.cuda(), mask.cuda()
        images_masked = im * (1 - mask) + mask
        output = model(torch.cat((im, mask), dim=1), mask)
        # pred_label = torch.argmax(output[0])
        train_loss = loss(output, label)
        train_loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]

        training_loss += train_loss.item()
        training_f1 += f1_score(label.data.cpu().numpy(), pred.cpu().numpy(), labels=np.arange(N_CLASS).tolist(),
                                average='macro', zero_division=0)

        if idx % 1000 == 0:
            print("Step {}: train loss - {:.03f} / train f1 - {:.03f}".format(idx, train_loss / 100, training_f1 / 100))
            training_f1 = 0
            training_loss = 0
    print("epoch {} complete".format(epoch + 1))
    torch.save({'svm': model.fc.state_dict()}, "{}/svm_weights.pth".format(SAVE_PATH))
    torch.save({'opt': optimizer.state_dict()}, "{}/opt_weights.pth".format(SAVE_PATH))
