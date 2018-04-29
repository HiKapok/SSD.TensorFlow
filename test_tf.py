import tensorflow as tf


b=tf.one_hot([-1,-2,0,1], 5, on_value=1, off_value=0, axis=1, dtype=tf.int32)

a=tf.split(tf.constant([[1,2,3,4],[11,12,13,14]]), 4, axis=1)

index = 0
t=tf.constant([0,1])
max_attempt = 3

def condition(index, t):
    return False

def body(index, t):
    return index+1, t

[index, t] = tf.while_loop(condition, body, [index, t], parallel_iterations=16, back_prop=False, swap_memory=True)

sess = tf.Session()
print(sess.run(b))
print(sess.run(index))
