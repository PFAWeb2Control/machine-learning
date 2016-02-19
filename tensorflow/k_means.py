import tensorflow as tf
import numpy as np
import time

K=5
MAX_ITERS = 1000

def clusterize(batch):
    start = time.time()
    
    n = len(batch)
    dim = len(batch[0])
    print batch
    points = tf.placeholder(tf.int32, [n,dim])
    cluster_assignments = tf.Variable(tf.zeros([n], dtype=tf.int64))

    # Use  K random points as the starting centroids
    centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0,0], [K,dim]))


    # Replicate to n copies of each centroid and K copies of each
    # point, then subtract and compute the sum of squared distances.
    rep_centroids = tf.reshape(tf.tile(centroids, [n, 1]), [n, K, dim])
    rep_points = tf.reshape(tf.tile(points, [1, K]), [n, K, dim])
    sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), reduction_indices=2)

    # Use argmin to select the lowest-distance point
    best_centroids = tf.argmin(sum_squares, 1)
    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, 
                                                    cluster_assignments))

    def bucket_mean(data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    means = bucket_mean(points, best_centroids, K)

    # Do not write to the assigned clusters variable until after
    # computing whether the assignments have changed - hence with_dependencies
    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(
            centroids.assign(means),
            cluster_assignments.assign(best_centroids))

    changed = True
    iters = 0
    sess = tf.Session()
    sess.run(tf.initialize_all_variables(), feed_dict={points: batch})

    while changed and iters < MAX_ITERS:
        iters += 1
        [changed, _] = sess.run([did_assignments_change, do_updates], feed_dict={points: batch})

    [centers, assignments] = sess.run([centroids, cluster_assignments], feed_dict={points: batch})
    end = time.time()
    print ("Found in %.2f seconds" % (end-start)), iters, "iterations"

    return [centers, assignments]

if __name__ == "__main__":
    clusterize(tf.random_uniform([1000,20], maxval=200, dtype=tf.int32).eval(session=tf.Session()))
