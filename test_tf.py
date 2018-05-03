import tensorflow as tf

from numpy import random

print(random.randint(2))
print(random.uniform(1, 4))



cls_pred = tf.constant([[[0., 2., 3.], [3., 1., 2.]], [[11., 0., 13.], [23., 21., 32.]]])
negtive_mask = tf.constant([[True, True], [True, True]], dtype=tf.bool)
predictions_for_bg = tf.nn.softmax(cls_pred)[:, :, 0]

prob_for_negtives = tf.where(negtive_mask,
                       0. - predictions_for_bg,
                       # ignore all the positives
                       0. - tf.ones_like(predictions_for_bg))
topk_prob_for_bg, _ = tf.nn.top_k(prob_for_negtives, k=2)


score_at_k = tf.gather_nd(topk_prob_for_bg, tf.stack([tf.range(tf.shape(cls_pred)[0]), tf.constant([1, 0])], axis=-1))

sess = tf.Session()
print(sess.run([topk_prob_for_bg, score_at_k]))

print('eee')

selected_neg_mask = prob_for_negtives > topk_prob_for_bg[-1]




t = tf.constant([1, 2, 3])

_,a=tf.nn.top_k(t, k=2, sorted=True)

paddings = tf.constant([[1, 1]])
# 'constant_values' is 0.
# rank of 't' is 2.
aaa = tf.pad(t, paddings, "CONSTANT")

b=tf.stack([0, 5], axis=0)

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
print(sess.run(a))
print(sess.run(index))
