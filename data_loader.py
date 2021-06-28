import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
np.set_printoptions(threshold=100000)
class ImageFolder(data.Dataset):
	def __init__(self, root,image_size=(256,256),mode='train',augmentation_prob=0.4):
		"""Initializes image paths and preprocessing module."""
		self.root = root
		#print(self.root)
		# GT : Ground Truth
		self.GT_paths = root[:-1]+'_GT/'
		self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
		self.image_size = image_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Reads an image from a file and preprocesses it and returns."""
		image_path = self.image_paths[index]
		#print(image_path)
		filename = image_path.split('/')[-1]
		label = filename.split("_")[0]
		#print(filename)
		GT_path = self.GT_paths + '/' + filename

		image = Image.open(image_path)
		GT = Image.open(GT_path)
		#print("image:",image.size)
		#print("GT:",GT.size)
		#print(image.size)
		aspect_ratio = image.size[1]/image.size[0]

		Transform = []

		#ResizeRange = random.randint(300,320)
		#Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
		CropRange = random.randint(390, 410)
		Transform.append(T.CenterCrop((CropRange, CropRange)))
		Transform = T.Compose(Transform)

		image = Transform(image)
		GT = Transform(GT)
		Transform = []
		p_transform = random.random()

		if (self.mode == 'train') and p_transform <= self.augmentation_prob:
			RotationDegree = random.randint(0,3)
			RotationDegree = self.RotationDegree[RotationDegree]
			if (RotationDegree == 90) or (RotationDegree == 270):
				aspect_ratio = 1/aspect_ratio

			Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
						
			RotationRange = random.randint(-10,10)
			Transform.append(T.RandomRotation((RotationRange,RotationRange)))
			#CropRange = random.randint(250,270)
			#Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
			#Transform = T.Compose(Transform)
			Transform = T.Compose(Transform)
			image = Transform(image)
			GT = Transform(GT)

			ShiftRange_left = random.randint(0,20)
			ShiftRange_upper = random.randint(0,20)
			ShiftRange_right = image.size[0] - random.randint(0,20)
			ShiftRange_lower = image.size[1] - random.randint(0,20)
			image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
			GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))

			if random.random() < 0.5:
				image = F.hflip(image)
				GT = F.hflip(GT)

			if random.random() < 0.5:
				image = F.vflip(image)
				GT = F.vflip(GT)

			Transform = T.ColorJitter(brightness=0.2,contrast=0.2,hue=0.02)

			image = Transform(image)

			Transform =[]


		Transform.append(T.Resize(self.image_size))
		Transform.append(T.ToTensor())
		Transform = T.Compose(Transform)
		
		image = Transform(image)
		#print(image.shape)
		GT = Transform(GT)
		GT1 = torch.unsqueeze(GT[0,:,:], dim=0)
		#Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		#image = Norm_(image)
		label_np = np.array(float(label))
		label_tensor = torch.from_numpy(label_np)
		return image, GT1, label_tensor,filename

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
