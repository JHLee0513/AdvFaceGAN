import sys
import torch
from tqdm import tqdm
from torchvision import transforms
from train import INPUT_SIZE, percent_gx

sys.path.append('../')
from models.generative import Generative
from models.digits import DigitModel
from data.dataloader import get_lfw_datasets_test
total = 0
correct = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
samples = test_set = get_lfw_datasets_test("../data/valid_set")

digits = DigitModel().to(device)
digits.load_state_dict(torch.load("../data/models/digits/digits_best.pth", map_location=device))
digits.eval()

for dummy_ten in tqdm(iter(samples)):
    GENERATOR_PATH = "../data/models/generator.pth" # change to path of prefered generator
    gen = Generative(INPUT_SIZE, rgb = False).to(device)
    gen.load_state_dict(torch.load(GENERATOR_PATH, map_location=device))

    test_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])

    dummy_ten = dummy_ten.to(device).unsqueeze(0)

    orig = dummy_ten
    gx = (percent_gx * gen(dummy_ten))
    adv = gx + dummy_ten
#     plotimg = torch.cat((orig.detach().reshape(28, 28), gx.detach().reshape(28, 28), adv.detach().reshape(28, 28)), 1) 
#     plt.imshow(plotimg.cpu().detach(), cmap='gray')
#     plt.show()

    target = digits(adv)
    if target.argmax(keepdim=False, dim=-1).item() == 4:
        correct += 1
    
    total += 1
    
print(correct/total)