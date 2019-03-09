#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from tqdm import tqdm
import torch.utils.data
import random
import torch.optim as optim
import os
import torchvision.utils as vutils



# In[2]:


def prepare_facedataset():
    masks = []
    images = []
    for filename in tqdm(glob.iglob('data/*/*', recursive=True)):
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128)).astype(np.float32)/255.
        mask  = np.zeros((1, 128, 128), dtype=np.float32)
        mask[:, 50:38*2, 20*2:45*2] = 1
        image = np.moveaxis(image, -1, 0)
#         print(image.shape)
#         eyes  = image[50:38*2, 20*2:45*2]
        images.append(image)
        masks.append(mask)
#         plt.imshow(mask[0, :, :])
#         plt.imshow(np.moveaxis(image, 0, -1))
#         plt.imshow(np.moveaxis(mask, 0, -1)[:, :, 0])
#         plt.show()
    return np.asarray(images), np.asarray(masks)
#     print(np.max(image))
#         plt.imshow(image*mask)
#         plt.show()
# a,b = prepare_facedataset()


# In[3]:


# print(a.shape)
# print(b.shape)


# In[4]:


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = prepare_facedataset()

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):

        return self.data[0][idx], self.data[1][idx]


# In[5]:


import torch
from torch import nn, optim

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64*4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64 * 4, 64 * 2, 4, 2, 1, bias=False,),
            nn.Conv2d(64 * 2, 64 * 2, 4, 2, padding=1, bias=False, dilation=2),
            nn.Conv2d(64 * 2, 64 * 2, 4, 2, padding = 2, bias=False, dilation=4),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d( 64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64 * 2, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(64 * 2, 64 * 2, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(64 * 2, 64 * 2, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(64 * 2, 64 * 2, 4, 2, 1, bias=False),
 
            nn.ConvTranspose2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 3, 4, 2, 1, bias=False),
        )
    def forward(self, image):
        encoded = self.encoder(image)
        complete = self.decoder(encoded)
        return complete


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (64*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            Flatten(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.main(image)


# In[ ]:





# In[6]:


# data = torch.randn((3, 4, 128, 128))
# generator.forward(data).shape


# In[7]:


# discriminator.forward(torch.randn(3, 3, 128, 128)).shape


# In[8]:


def sample_images(iters, fake, nrow=4, test=""):
    if test == "":
        os.makedirs('result/dcgan/test', exist_ok=True)
        img = vutils.make_grid(fake[:nrow*nrow, ...], nrow=nrow, padding=2, normalize=True)
        plt.imshow(np.transpose(img,(1,2,0)) )
        plt.axis('off')
        plt.savefig("result/dcgan/test/%d%s.pdf" % (iters, test),  pad_inches = 0)
        plt.close()
    else:
        os.makedirs('result/dcgan/true_validation/', exist_ok=True)
        img = vutils.make_grid(fake[:nrow*nrow, ...], nrow=nrow, padding=2, normalize=True)
        plt.imshow(np.transpose(img,(1,2,0)) )
        plt.axis('off')
        plt.savefig("result/dcgan/true_validation/%s.pdf" % (test),  pad_inches = 0)
        plt.close()
# In[ ]:


manualSeed = 0
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu=1
beta1=0.99
lr=1e-4
num_epochs=1000

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)



dataset = FaceDataset()

dataset_portions = (10587, 1323, 1323)

train_data, valid_data, test_data = torch.utils.data.random_split(dataset, dataset_portions)
print(len(train_data))
print(len(valid_data))
print(len(test_data))

dataloader = torch.utils.data.DataLoader(train_data,
                                             batch_size=256, shuffle=True,
                                             num_workers=12, pin_memory=True)

val_dataloader = torch.utils.data.DataLoader(valid_data,
                                             batch_size=256, shuffle=True,
                                             num_workers=12, pin_memory=True)

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
loss = nn.BCELoss()


# In[ ]:



for val_face, val_mask in val_dataloader:
    val_face_gpu = val_face.to(device)
    val_mask_gpu = val_mask.to(device)
    val_face_masked_gpu_3_channel = val_face_gpu * val_mask_gpu
    val_face_masked_gpu_4_channel = torch.cat((val_face_masked_gpu_3_channel, val_mask_gpu), 1)
    break
sample_images(None, val_face_gpu.cpu(), test="valid_1")
mseloss = nn.MSELoss()
# In[ ]:


# For each epoch
iters = 0
real_label=1
fake_label=0
for epoch in tqdm(range(num_epochs)):
    # For each batch in the dataloader
    for i, (face, mask) in enumerate(dataloader):
        # print(data.shape)
        # egg = egg.to(device)
        # output_image = output_image.to(device)


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
#         plt.imshow(np.moveaxis(mask.numpy(), 1, -1)[0, :, :, 0])
#         plt.show()
        discriminator.zero_grad()
        face_gpu = face.to(device)
        mask_gpu = mask.to(device)
        face_masked_gpu_3_channel = face_gpu * mask_gpu
        face_masked_gpu_4_channel = torch.cat((face_masked_gpu_3_channel, mask_gpu), 1)

#         b_size = face_masked_gpu_4_channel.size(0)
#         label = torch.full((b_size,), real_label, device=device)
#         # Forward pass real batch through D
#         output = discriminator(face_gpu).view(-1)
# #         print(output.shape)
# #         print(label.shape)
#         err_D_real = loss(output, label)
#         err_D_real.backward()
#         D_x = output.mean().item()



#         ## Train with all-fake batch
#         # Generate batch of latent vectors
# #         noise = torch.randn(b_size, 100, 1, 1, device=device)
#         # Generate fake image batch with G
        fake = generator(face_masked_gpu_4_channel)
#         label.fill_(fake_label)
#         # Classify all fake batch with D
#         output = discriminator(fake.detach()).view(-1)
#         # Calculate D's loss on the all-fake batch
#         err_D_fake = loss(output, label)
#         # Calculate the gradients for this batch
#         err_D_fake.backward()
#         D_G_z1 = output.mean().item()
#         # Add the gradients from the all-real and all-fake batches
#         err_D = err_D_real + err_D_fake

#         # Update D
#         optimizerD.step()
# #         generator_train_count+=1


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        # if generator_train_count % generator_train_frequency == 0:
        # for _ in range(generator_train_frequency):
        generator.zero_grad()
        # label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        # output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        mse_err = mseloss(fake, face_gpu)
        err_G = mse_err
        # Calculate gradients for G
        err_G.backward()
        # Update G
        # D_G_z2 = output.mean().item()
        optimizerG.step()
        # print(i)
        err_G_loss = err_G.item()

        # Output training stats
        if i % 50 == 0 :
            print('[%d/%d][%d/%d]\tLoss_G: %.4f\t'
                  % (epoch, num_epochs, i, len(dataloader),
                    err_G_loss))
            # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            #       % (epoch, num_epochs, i, len(dataloader),
            #          err_D.item(), err_G_loss, D_x, D_G_z1, D_G_z2))

        if (iters % 50 == 0):
            with torch.no_grad():
                fake = generator(val_face_masked_gpu_4_channel).detach().cpu()
            sample_images(iters, fake)
        iters+=1    
os.makedirs('result/dcgan/model/', exist_ok=True)
torch.save(generator.state_dict(), 'result/dcgan/model/gen.pth')
torch.save(discriminator.state_dict(), 'result/dcgan/model/dis.pth')



# In[ ]:


device


# In[ ]:


torch.cat((torch.randn(3, 3, 128, 128), torch.randn(3,1,128,128)), 1).shape


# In[ ]:




