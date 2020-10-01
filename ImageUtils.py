from PIL import Image
import numpy as np
import tensorflow as tf
import random
"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
	"""Parse a record to an image and perform data preprocessing.

	Args:
		record: An array of shape [3072,]. One row of the x_* matrix.
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	# Reshape from [depth * height * width] to [depth, height, width].
	# depth_major = tf.reshape(record, [3, 32, 32])
	depth_major = record.reshape((3, 32, 32))

	# Convert from [depth, height, width] to [height, width, depth]
	# image = tf.transpose(depth_major, [1, 2, 0])
	image = np.transpose(depth_major, [1, 2, 0])

	image = preprocess_image(image, training)

	return image


def preprocess_image(image, training):
	"""Preprocess a single image of shape [height, width, depth].

	Args:
		image: An array of shape [32, 32, 3].
		training: A boolean. Determine whether it is in training mode.

	Returns:
		image: An array of shape [32, 32, 3].
	"""
	temp_width = 32+8
	width = 32
	temp_height = 32+8
	height = 32
	if training:
		### YOUR CODE HERE
		# Resize the image to add four extra pixels on each side.
		# image2 = tf.image.resize_image_with_crop_or_pad(image, temp_height, temp_width)
		
		# width_diff = temp_width - width
		# offset_crop_width = int(max(-width_diff // 2, 0))
		# offset_pad_width = int(max(width_diff // 2, 0))

		# height_diff = temp_height - height
		# offset_crop_height = int(max(-height_diff // 2, 0))
		# offset_pad_height = int(max(height_diff // 2, 0))

		# image = image[offset_crop_height:offset_crop_height+min(temp_height, height),offset_crop_width:offset_crop_width+min(temp_width, width), :]
		# image = np.pad(image, pad_width=[(offset_pad_height, offset_pad_height),(offset_pad_width, offset_pad_width), (0, 0)], mode='constant')
		image_padded=np.zeros((40,40,3))
		image_padded[4:36,4:36] = image

		### END CODE HERE

		### YOUR CODE HERE
		# Randomly crop a [32, 32] section of the image.
		# image = tf.random_crop(image, [32, 32, 3])

		# HINT: randomly generate the upper left point of the image
		# start_height = int((random.random()*10000))%height_diff
		# start_width = int(random.random()*10000)%width_diff
		# image = image[start_height:start_height+height, start_width:start_width+width, :]
		start_height=random.randint(0,8)
		start_width=random.randint(0,8)
		image=image_padded[start_height:start_height+32,start_width:start_width+32]
		### END CODE HERE

		### YOUR CODE HERE
		# Randomly flip the image horizontally.
		# image = tf.image.random_flip_left_right(image)
		chance = round(random.random())
		if chance >=0.5:
			image = np.fliplr(image)

		### END CODE HERE

	### YOUR CODE HERE
	# Subtract off the mean and divide by the standard deviation of the pixels.
	# image = tf.image.per_image_standardization(image)
	
	stddev = np.std(image)
	mean = np.mean(image)
	N= np.size(image)
	adjusted_stddev = max(stddev, 1.0/(N**(0.5)))
	image = (image -mean) / adjusted_stddev	

	#image = (image - np.mean(image)) / np.std(image)
	### END CODE HERE

	return image

