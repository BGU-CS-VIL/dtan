import tensorflow as tf
from DTAN.smoothness_prior import smoothness_norm


def alignment_loss(X,loss_type ='l2'):
    ''' Alignment loss
            Keras loss function with TensorFlow backed. Calculates the within class variance and returns the MSE
            Args:
            X - Batch training data - STN output
            y_true, y_pred - class label. Required for Keras loss function
            [TBD] reference - instead of calculating withing class congealing loss, calculates MSE from reference signal.

            Returns:
            Alignment loss

    '''
    class alignment_class:
        def __init__(self, X, classes, class_idx, y_true, loss_type = loss_type):
            self.X = tf.cast(X, tf.float32)
            self.classes = tf.cast(classes, tf.float32)
            self.class_idx = tf.cast(class_idx, tf.float32)
            self.y_true = tf.cast(y_true, tf.float32)
            self.loss_type = loss_type
            assert self.loss_type in ['l1', 'l2'], "Alignment loss type must be: l1, l2"

        def l1_alignment_loss(self, X_within_class):
            # calculate within class variance

            mean, _ = tf.nn.moments(X_within_class, axes=[0])

            # Reshaping: [batch size,dim, channels]
            mean_tile = tf.tile(tf.expand_dims(mean, 0), [tf.shape(X_within_class)[0],1,1])
            # should calculate from median signal?
            alignment_loss = tf.losses.absolute_difference(X_within_class,mean_tile)
            return alignment_loss

        def l2_alignment_loss(self, X_within_class):
            # calculate within class variance

            _, variance = tf.nn.moments(X_within_class, axes=[0])
            # might fix Nan issue...
            #variance = tf.Print(variance, [variance], message="variance:")
            alignment_loss = tf.reduce_mean(variance)
            return alignment_loss

        def alignment_loss_per_class(self, current_class):
            # Slice X within class
            # classes, class_idx = tf.unique(y) #not one hot
            # convert from one-hot
            # y = tf.argmax(self.y_true,axis=1)
            y = tf.squeeze(self.y_true)

            # create classes range vector for classes in batch
            classes, class_idx = tf.unique(y)  # not one hot

            X_within_class = tf.gather(self.X, tf.where(tf.equal(class_idx, tf.cast(current_class, tf.int32))))
            X_within_class = tf.cast(X_within_class, tf.float32)



            if self.loss_type == 'l1':
                align_loss_within_class = self.l1_alignment_loss(X_within_class)
            else :
                align_loss_within_class = self.l2_alignment_loss(X_within_class)
            # Due to small batch size, some classes might not appear and loss might be nan:
            def is_nan(): return tf.keras.backend.epsilon()
            def is_not_nan(): return align_loss_within_class

            ans = tf.cond(tf.is_nan(align_loss_within_class), is_nan, is_not_nan)
            return ans


    def loss_func(y_true, y_pred):
        # convert from one-hot
        # y = tf.argmax(y_true,axis=1)
        y = tf.cast(y_true, tf.float32)
        y = tf.squeeze(y)

        # create classes range vector for classes in batch
        classes, class_idx = tf.unique(y)  # not one hot

        alg = alignment_class(X, classes, class_idx, y_true)

        # Get within class MSE
        loss = tf.map_fn(alg.alignment_loss_per_class, classes)
        loss = tf.reduce_mean(loss)
        #
        return loss

    return loss_func


def alignment_loss_with_prior(signal, T, theta, scale_spatial=1, scale_value=1 ):
    loss = alignment_loss(signal)
    smoothness_loss = smoothness_norm(T, theta, scale_spatial, scale_value)
    loss_with_prior = tf.add(loss, smoothness_loss)

    return loss_with_prior
