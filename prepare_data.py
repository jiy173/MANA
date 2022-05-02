import argparse
import h5py
import numpy as np
import PIL.Image as pil_image

parser = argparse.ArgumentParser(description='Indicate Vimeo90k dataset path.')
parser.add_argument('--dataset', help='Vimeo90K dataset path (the folder containing both sequences/ and seq_trainlist.txt)')
parser.add_argument('--output', help='Output h5 file path')
args = parser.parse_args()

root=args.dataset
output=args.output
trainlist=np.genfromtxt(root+'seq_trainlist.txt',dtype='str')
pre_hr=root+'sequences/'


h5_file = h5py.File(output, 'w')

lr_group = h5_file.create_group('lr')
hr_group = h5_file.create_group('hr')

patch_idx = 0
for i, vid_path in enumerate(trainlist):
    hr=np.zeros((7,3,256,448)).astype(np.uint8)
    lr=np.zeros((7,3,64,112)).astype(np.uint8)
    for k in range(7):
        hr_temp=pil_image.open(pre_hr+vid_path+'/im'+str(k+1)+'.png')
        lr_temp=hr_temp.resize((hr_temp.size[0]//4,hr_temp.size[1]//4), pil_image.BICUBIC)
        hr[k,:,:,:]=np.asarray(hr_temp).astype(np.uint8).transpose(2,0,1)
        lr[k,:,:,:]=np.asarray(lr_temp).astype(np.uint8).transpose(2,0,1)
   
    lr_group.create_dataset(str(patch_idx), data=lr)
    hr_group.create_dataset(str(patch_idx), data=hr)
    
    patch_idx += 1
    print(patch_idx)

h5_file.close()