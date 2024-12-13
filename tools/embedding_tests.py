import scipy
import numpy as np
import torch
from torchvision.models import vit_b_16
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

def  run_test():
    data = scipy.io.loadmat('../examples/Burgers_Equation_1d/burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]

    Exact = torch.Tensor(np.real(data['usol']).T).float().to('cuda')

    Exact = Exact.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    vit_model = vit_b_16(pretrained=True).to('cuda')
    vit_model.eval()  # Set to evaluation mode
    vit_transform = Compose([
        Resize((224, 224)),  # Resize to ViT input size

        # ImageNet normalization (if using pretrained ViT)
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Simple [-1,1] normalization (for custom data)
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    with torch.no_grad():
        u_i_image = vit_transform(Exact)
        vit_embedding = vit_model(u_i_image)  # Extract ViT embeddings

    print("mu")


if __name__ == "__main__":
    run_test()