


import tensorflow as tf
import numpy as np



inputs = np.array([[0,0],[1,1],[2,2],[3,3],[4,4]])
second = np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])

out = np.array([[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])
out = tf.convert_to_tensor(out,dtype=tf.float32)


inputs = tf.convert_to_tensor(inputs,dtype=tf.float32)
second = tf.convert_to_tensor(second,dtype=tf.float32)





tarr = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True,infer_shape=False)
tarr = tarr.write(0,inputs)
tarr = tarr.write(1,out)


#initializer as first element

def function(past,new):

	(z,y) = new
	(z_prev,y_prev) = past

	return (z_prev+z,y + 3)



out = tf.scan(function,(inputs,second))

with tf.Session() as sess:
	z,y=sess.run(out)
	print(z)
	print(y)
	tar = sess.run(tarr.read(0))
	tar2 = sess.run(tarr.read(1))
	print(tar)
	print(tar2)

	