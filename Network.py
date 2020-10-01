import tensorflow as tf

"""This script defines the network.
"""

class ResNet(object):

	def __init__(self, resnet_version, resnet_size, num_classes, 
					first_num_filters):
		"""Define hyperparameters.

		Args:
			resnet_version: 1 or 2. If 2, use the bottleneck blocks.
			resnet_size: A positive integer (n).
			num_classes: A positive integer. Define the number of classes.
			first_num_filters: An integer. The number of filters to use for the
				first block layer of the model. This number is then doubled
				for each subsampling block layer.
		"""
		self.resnet_version = resnet_version
		self.resnet_size = resnet_size
		self.num_classes = num_classes
		self.first_num_filters = first_num_filters

	def __call__(self, inputs, training):
		"""Classify a batch of input images.

		Architecture (first_num_filters = 16):
		layer_name      | start | stack1 | stack2 | stack3 | output
		output_map_size | 32x32 | 32×32  | 16×16  | 8×8    | 1x1
		#layers 		| 1 	| 2n/3n	 | 2n/3n  | 2n/3n  | 1
		#filters 		| 16 	| 16(*4) | 32(*4) | 64(*4) | num_classes

		n = #residual_blocks in each stack layer = self.resnet_size
		The standard_block has 2 layers each.
		The bottleneck_block has 3 layers each.

		Example of replacing:
		standard_block 		conv3-16 + conv3-16
		bottleneck_block	conv1-16 + conv3-16 + conv1-64

		Args:
			inputs: A Tensor representing a batch of input images.
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			A logits Tensor of shape [<batch_size>, self.num_classes].
		"""
		
		outputs = self._start_layer(inputs, training)

		if self.resnet_version == 1:
			block_fn = self._standard_block_v1
		else:
			block_fn = self._bottleneck_block_v2

		for i in range(3):
			filters = self.first_num_filters * (2**i)
			strides = 1 if i == 0 else 2
			outputs = self._stack_layer(outputs, filters, block_fn, strides, training)

		outputs = self._output_layer(outputs, training)

		return outputs

	################################################################################
	# Blocks building the network
	################################################################################
	def _batch_norm_relu(self, inputs, training):
		"""Perform batch normalization then relu."""

		### YOUR CODE HERE
		outputs = tf.layers.batch_normalization(inputs, training = training)
		outputs = tf.nn.relu(outputs)
		### END CODE HERE

		return outputs

	def _start_layer(self, inputs, training):
		"""Implement the start layer.

		Args:
			inputs: A Tensor of shape [<batch_size>, 32, 32, 3].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A Tensor of shape [<batch_size>, 32, 32, self.first_num_filters].
		"""
		
		### YOUR CODE HERE
		#initial conv
		outputs = self._conv2d(inputs=inputs,filters=self.first_num_filters,kernel_size=3,strides=1,training=training)

		### END CODE HERE

		# We do not include batch normalization or activation functions in V2
		# for the initial conv1 because the first block unit will perform these
		# for both the shortcut and non-shortcut paths as part of the first
		# block's projection.
		if self.resnet_version == 1:
			outputs = self._batch_norm_relu(outputs, training)

		return outputs

	def _output_layer(self, inputs, training):
		"""Implement the output layer.

		Args:
			inputs: A Tensor of shape [<batch_size>, 8, 8, channels].
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: A logits Tensor of shape [<batch_size>, self.num_classes].
		"""

		# Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
		if self.resnet_version == 2:
			inputs = self._batch_norm_relu(inputs, training)

		### YOUR CODE HERE
		inputs = tf.reduce_mean(input_tensor=inputs, axis=[1, 2])
		inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
		if not training:
			inputs= tf.nn.softmax(inputs)
		outputs = inputs
		### END CODE HERE

		return outputs

	def _stack_layer(self, inputs, filters, block_fn, strides, training):
		"""Creates one stack of standard blocks or bottleneck blocks.
		
		Args:
			inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
			filters: A positive integer. The number of filters for the first
				convolution in a block.
			block_fn: 'standard_block' or 'bottleneck_block'.
			strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
			training: A boolean. Used by operations that work differently
				in training and testing phases.

		Returns:
			outputs: The output tensor of the block layer.
		"""

		filters_out = filters * 4 if self.resnet_version == 2 else filters
		
		num_res_blocks = self.resnet_size
		def projection_shortcut(inputs):
			### YOUR CODE HERE
			inputs = self._conv2d(
					inputs=inputs,
					filters=filters_out,
					kernel_size=1,
					strides=strides,
					training=training)
			return inputs
			### END CODE HERE

		### YOUR CODE HERE
		# Only the first block per stack_layer uses projection_shortcut
		inputs = block_fn(
		      inputs=inputs,
		      filters=filters,
		      training=training,
		      projection_shortcut=projection_shortcut,
		      strides=strides)

		for i in range(1, num_res_blocks):
		    inputs = block_fn(
		        inputs=inputs,
		        filters=filters,
		        training=training,
		        projection_shortcut=None,
		        strides=1)

		outputs = inputs
		### END CODE HERE

		return outputs

	def _standard_block_v1(self, inputs, filters, training, projection_shortcut, strides):
		"""Creates a standard residual block for ResNet v1.
		
		Args:
			inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
			filters: A positive integer. The number of filters for the first
				convolution.
			training: A boolean. Used by operations that work differently
				in training and testing phases.
			projection_shortcut: The function to use for projection shortcuts
      			(typically a 1x1 convolution when downsampling the input).
			strides: A positive integer. The stride to use for the block. If
				greater than 1, this block will ultimately downsample the input.

		Returns:
			outputs: The output tensor of the block layer.
		"""
		shortcut = inputs

		if projection_shortcut is not None:
			### YOUR CODE HERE
			shortcut = projection_shortcut(inputs)
			#shortcut = tf.layers.batch_normalization(shortcut, training=training)
			### END CODE HERE

		### YOUR CODE HERE
		inputs = self._conv2d(
				inputs=inputs,
				filters=filters,
				kernel_size=3,
				strides=strides,
				training=training)

		inputs = self._batch_norm_relu(inputs, training)
		inputs = self._conv2d(
				inputs=inputs,
				filters=filters,
				kernel_size=3,
				strides=1,
				training=training)

		inputs = tf.layers.batch_normalization(inputs, training=training)
		outputs = inputs + shortcut
		outputs = tf.nn.relu(outputs)

		### END CODE HERE

		return outputs

	def _bottleneck_block_v2(self, inputs, filters, training, projection_shortcut, strides):
		"""Creates a bottleneck block for ResNet v2.
		
		Args:
			inputs: A Tensor of shape [<batch_size>, height_in, width_in, channels].
			filters: A positive integer. The number of filters for the first
				convolution. NOTE: filters_out will be 4xfilters.
			training: A boolean. Used by operations that work differently
				in training and testing phases.
			projection_shortcut: The function to use for projection shortcuts
      			(typically a 1x1 convolution when downsampling the input).
			strides: A positive integer. The stride to use for the block. If
				greater than 1, this block will ultimately downsample the input.

		Returns:
			outputs: The output tensor of the block layer.
		"""

		### YOUR CODE HERE
		shortcut = inputs
		inputs = self._batch_norm_relu(inputs, training)

		# The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
		if projection_shortcut is not None:
			shortcut = projection_shortcut(inputs)

		inputs = self._conv2d(inputs=inputs, filters=filters, kernel_size=1, strides=1, training=training)

		inputs = self._batch_norm_relu(inputs, training)
		inputs = self._conv2d(inputs=inputs, filters=filters, kernel_size=3, strides=strides, training=training)

		inputs = self._batch_norm_relu(inputs, training)
		inputs = self._conv2d(inputs=inputs, filters=filters*4, kernel_size=1, strides=1, training=training)

		outputs = inputs + shortcut
		### END CODE HERE

		return outputs

	def _conv2d(self,inputs,filters,kernel_size,strides,training=None):
		conv = tf.layers.Conv2D(
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding="SAME",
			trainable = training)
		y=conv(inputs)
		return y


