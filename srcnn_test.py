'''
Test
'''

from __future__ import print_function
import argparse
import torch
import math
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np
import os

#Settings
#Totally 5 test images you can use in this foler.
mode1 = 'SRCNN-SC'
mode2 = 'SRCNN-TC'
mode3 = 'VDSR'
model1 = 'checkpoint\SSmodel915_epoch_470.pth'
model2 = 'checkpoint\Smodel915_epoch_370.pth'
model3 = 'checkpoint\Vmodel_epoch_620.pth'
mode = mode3

for i in range(5):
    input_image = './set5/set5/img_00'+str(i+1)+'.png'
    model = model3
    # model = 'checkpoint\SSmodel915_epoch_470.pth'    #your model dir
    output_filename = 'output/'+mode+'/output_'+str(i+1)+'.png'    #your output dir
    scale_factor = 3
    use_cuda = 1

    img1 = Image.open(input_image)


    max_size0 = img1.size[0] - (img1.size[0] % scale_factor)
    max_size1 = img1.size[1] - (img1.size[1] % scale_factor)
    img = img1.crop((0,0,max_size0,max_size1))

    img2 = img.copy()

    img2 = img2.resize((int(img2.size[0]//scale_factor),int(img2.size[1]//scale_factor)),Image.BICUBIC)
    img2 = img2.resize((int(img2.size[0]*scale_factor),int(img2.size[1]*scale_factor)),Image.BICUBIC)

    #y, cb, cr = img2.convert("YCbCr").split()
    #print(y.size)
    y = img2

    model = torch.load(model)
    input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])

    if use_cuda:
        model = model.cuda(device = 0)
        input = input.cuda(device = 0)

    out = model(input)
    out = out.cpu()

    # print ("type = ",type(out))
    tt = transforms.ToPILImage()

    img_out = tt(out.data[0])
    img2.save('output/'+mode+"/input"+str(i+1)+".png")
    img_out.save(output_filename)




    # out_img_y = out.data[0].numpy()
    # out_img_y *= 255.0
    # out_img_y = out_img_y.clip(0, 255)
    # out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    # img_out = Image.merge('YCbCr', [out_img_y, cb, cr]).convert('RGB')
    
    # img2.save('output/'+mode+"/input"+str(i+1)+".png")
    # img_out.save(output_filename)
    
    img_out = img_out.convert('F')
    img2 = img2.convert('F')
    img = img.convert('F')
    
    mse = ((np.array(img)-np.array(img_out))**2).mean()
    psnr = 10*math.log10(255*255/mse)
    print("PSNR of"+" "+str(i+1)+" "+":",psnr)
    
    mse2 = ((np.array(img)-np.array(img2))**2).mean()
    psnr2 = 10*math.log10(255*255/mse2)
    print("PSNR2 of"+" "+str(i+1)+" "+":",psnr2)