# FIX THIS: rename however you want

class Discriminative(nn.Module):
    def __init__(self, image_size):
        super(Generative, self).__init__()
        if (image_size % 4 != 0):
            raise Exception("Image must be a factor of 4")
        # in_channels, num filters, kernal_size

    def forward(self, x):
        x = torch.GlobalAveragePooling2d(1,1)
        x = nn.Linear(3, 1)
        return torch.sigmoid(x)



        
