import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

image = torch.rand(1, 1, 28, 28) 

plt.plot(image.view(-1).numpy())
plt.title('Pixel Values of the Image')
plt.xlabel('Pixel Index')
plt.ylabel('Pixel Value')
plt.show()

print("Image Tensor:")

conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)

print("\nConvolutional Layer:")
print(conv1)

output=conv1(image)

print("Output: ",output)

print(output.shape)

num_channels = output.size(1) 

fig, axes = plt.subplots(1, num_channels, figsize=(12, 4))
for i in range(num_channels):
    axes[i].imshow(output[0, i].detach().numpy(), cmap='gray')
    axes[i].set_title(f'Channel {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()