import datetime
import os
import numpy as np
import tensorflow as tf
import glob
import cv2
import numpy as np
import random
import model
import tensorflow as tf

import matplotlib.pyplot as plt
# tf.compat.v1.disable_eager_execution()
# from tensorflow.examples.tutorials.mnist import mnist

patch_size = 48
image_size = 1024
# _, ax1 = plt.subplots(1, 1)
# fig, ax = plt.subplots(ncols=2, nrows=1)
# plt.ion()
def crop(img_pair, shape):
    # patch_size = .patch_size
    O_list = []
    B_list = []
    for img, (h, ww, _) in zip(img_pair, shape):
    # h, ww, _ = shape
        h_pad = image_size - h
        w_pad = image_size - ww
        # print(h, ww, h_pad, w_pad)
        # ax1.imshow(img)

        img = img[h_pad:, w_pad:, :]

        h, ww, c = img.shape

        w = ww // 2
        p_h = p_w = patch_size

        # ax[0].imshow(img[:, :w, :])
        # ax[1].imshow(img[:, w+1:ww, :])
        # plt.pause(1000)
        # plt.show()

        # if aug:
        #     mini = - 1 / 4 * patch_size
        #     maxi = 1 / 4 * patch_size + 1
        #     p_h = patch_size + self.rand_state.randint(mini, maxi)
        #     p_w = patch_size + self.rand_state.randint(mini, maxi)
        # else:
        #     p_h, p_w = patch_size, patch_size
        #
        # r = self.rand_state.randint(0, h - p_h)
        # c = self.rand_state.randint(0, w - p_w)
        r = random.randrange(0, h - p_h + 1)
        c = random.randrange(0, w - p_w + 1)

        # O = img_pair[:, w:]
        # B = img_pair[:, :w]
        O_list.append(img[r: r + p_h, c + w: c + p_w + w, :])  # rain 右边
        B_list.append(img[r: r + p_h, c: c + p_w, :])  # norain 左边
        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)
        # ax[0].imshow(B_list[-1])
        # ax[1].imshow(O_list[-1])
        # print(O_list[-1].shape, B_list[-1].shape)
        # plt.pause(1)
        # plt.show()
    # if aug:
    #     O = cv2.resize(O, (patch_size, patch_size))
    #     B = cv2.resize(B, (patch_size, patch_size))

    return np.stack(O_list, axis=0),  np.stack(B_list, axis=0)

def decode(serialized_example):
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'shape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
          'orishape': tf.FixedLenFeature(shape=(3,), dtype=tf.int64),
      })

  #
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image = tf.reshape(image, features['shape'])
  # height =
  # image.set_shape((mnist.IMAGE_PIXELS))
  # label = tf.cast(features['label'], tf.int32)
  return image, features['orishape']



def inputs(tfrecords_path, patch_size, batch_size, num_epochs):
    if not num_epochs:
        num_epochs = None
    with tf.name_scope("input"):
        #读取tfrecords文件
        dataset = tf.data.TFRecordDataset(tfrecords_path)
        #tfrecords数据解码
        dataset = dataset.map(decode)
        # dataset = dataset.map(crop)
        # dataset = dataset.map(normalize)
        #打乱数据的顺序
        dataset = dataset.shuffle(1000 + 3 * batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()



# model_path = './TrainedModel/model-Rain200L' # trained model

input_path = './TestImg/rainL/' # the path of testing images

results_path = './TestImg/resultsL/'  # the path of de-rained results



def _parse_function(filename):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string, channels=3)
  rain = tf.cast(image_decoded, tf.float32)/255.
  return rain

# tf_image_batch, tf_shape = inputs("./rain100L", patch_size=100, batch_size=10, num_epochs=300)
# with tf.Session() as sess:
#     image_batch, shape = sess.run([tf_image_batch, tf_shape])
#     print(image_batch.shape, shape)

if __name__ == '__main__':

    tf.reset_default_graph()
    epochs = 101
    train_batch_size = 32  # training batch size
    test_batch_size = 16  # validation batch size
    # image_size = 100  # patch size
    # batch_size = 8
    iterations = 10000 // train_batch_size  # total number of iterations to use.
    model_directory = './models_100L'  # directory to save trained model to.
    data_dir = '../../derain/dataset/rain100L'  # training data
    # test_data_name = '../../derain/dataset/rain100L/test'  # validation data
    # restore = False  # load model or not
    method = 'Adam'  # training method: Adam or SGD
    model_path = './models_100L/19'  # trained model ./TrainedModel/model-Rain200L
    ############## loading data
    # train_data = sio.loadmat(train_data_name)
    # test_data = sio.loadmat(test_data_name)
    tf_image_batch, tf_shape = inputs("./rain100L", patch_size=patch_size, batch_size=train_batch_size, num_epochs=100)
    # train_data = h5py.File(train_data_name)  # for large data ( v7.3 data)
    # train_data = train_data['feature_data'][:]

    # test_data = h5py.File(test_data_name)
    # test_data = test_data['feature_data'][:]

    ############## placeholder for training
    gt = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 3])
    inputs = tf.placeholder(dtype=tf.float32, shape=[train_batch_size, patch_size, patch_size, 3])

    ############# placeholder for testing
    # test_gt = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, 8])
    #
    # test_lms = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, 8])
    # test_ms_hp = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size // 4, image_size // 4, 8])
    # test_pan_hp = tf.placeholder(dtype=tf.float32, shape=[test_batch_size, image_size, image_size, 1])

    ######## network architecture
    outputs = model.Inference(inputs)
    # mrs = tf.add(mrs, lms)  # last in the architecture: add two terms together

    # Test
    # test_rs = DMDNet(test_ms_hp, test_pan_hp, reuse=True)
    # test_rs = test_rs + test_lms  # same as: test_rs = tf.add(test_rs,test_lms)

    ######## loss function
    l1 = tf.losses.absolute_difference(gt, outputs)#tf.reduce_mean(tf.abs(outputs - gt))
    # test_mse = tf.reduce_mean(tf.square(test_rs - test_gt))

    ##### Loss summary
    l1_loss_sum = tf.summary.scalar("l1_loss", l1)

    # test_mse_sum = tf.summary.scalar("test_loss", test_mse)

    # lms_sum = tf.summary.image("lms", tf.clip_by_value(vis_ms(lms), 0, 1))
    # mrs_sum = tf.summary.image("rs", tf.clip_by_value(vis_ms(mrs), 0, 1))

    # label_sum = tf.summary.image("label", tf.clip_by_value(vis_ms(gt), 0, 1))

    # all_sum = tf.summary.merge([mse_loss_sum, mrs_sum, label_sum, lms_sum])
    all_sum = tf.summary.merge([l1_loss_sum])

    #########   optimal    Adam or SGD

    t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    print(t_vars)
    if method == 'Adam':
        g_optim = tf.train.AdamOptimizer(0.0001) \
            .minimize(l1, var_list=t_vars)

    else:
        global_steps = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.1, global_steps, decay_steps=50000, decay_rate=0.1)
        clip_value = 0.1 / lr
        optim = tf.train.MomentumOptimizer(lr, 0.9)
        gradient, var = zip(*optim.compute_gradients(l1, var_list=t_vars))
        gradient, _ = tf.clip_by_global_norm(gradient, clip_value)
        g_optim = optim.apply_gradients(zip(gradient, var), global_step=global_steps)

    ##### GPU setting
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #### Run the above

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)

        if os.path.exists(model_path):
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     print("loading failure")
        #     raise EOFError
        #### read training data #####
        # gt1 = train_data['gt'][...]  ## ground truth N*H*W*C
        # pan1 = train_data['pan'][...]  #### Pan image N*H*W
        # ms_lr1 = train_data['ms'][...]  ### low resolution MS image
        # lms1 = train_data['lms'][...]  #### MS image interpolation to Pan scale



        # N = gt1.shape[0]

        #### read validation data #####
        # gt2 = test_data['gt'][...]  ## ground truth N*H*W*C
        # pan2 = test_data['pan'][...]  #### Pan image N*H*W
        # ms_lr2 = test_data['ms'][...]  ### low resolution MS image
        # lms2 = test_data['lms'][...]  #### MS image interpolation -to Pan scale

        # gt2 = np.array(gt2, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
        # pan2 = np.array(pan2, dtype=np.float32) / 2047.
        # ms_lr2 = np.array(ms_lr2, dtype=np.float32) / 2047.
        # lms2 = np.array(lms2, dtype=np.float32) / 2047.
        # N2 = gt2.shape[0]

        mse_train = []
        mse_valid = []

        start = datetime.datetime.now()
        start_epoch = 0 if model_path == '' else int(model_path.split('/')[-1])+1
        for epoch in range(start_epoch, epochs+1):
            for i in range(1, iterations+1):
                ###################################################################
                #### training phase! ###########################

                # bs = train_batch_size
                # batch_index = np.random.randint(0, N, size=bs)

                # train_gt = gt1[batch_index, :, :, :]
                # pan_batch = pan1[batch_index, :, :]
                # ms_lr_batch = ms_lr1[batch_index, :, :, :]
                # train_lms = lms1[batch_index, :, :, :]
                image_batch, image_shape = sess.run([tf_image_batch, tf_shape])
                image_batch_b = image_batch / 255.
                # print(image_batch_b.shape)
                O, B = crop(image_batch_b, image_shape)
                # print(O.shape, B.shape)
                # O = np.array(O, dtype=np.float32)
                # B = np.array(B, dtype=np.float32)
                # pan_hp_batch = get_edge(pan_batch)
                # train_pan_hp = pan_hp_batch[:, :, :, np.newaxis]  # expand to N*H*W*1

                # train_ms_hp = get_edge(ms_lr_batch)

                # train_gt, train_lms, train_pan_hp, train_ms_hp = get_batch(train_data, bs = train_batch_size)

                _, l1_loss, merged = sess.run([g_optim, l1, all_sum],
                                                feed_dict={gt: B, inputs: O})
                #
                mse_train.append(l1_loss)  # record the mse of trainning
                #
                if i % 10 == 0:
                    print("Epoch: " + str(epoch) + " Iter: " + str(i) + " l1: " + str(l1_loss))  # print, e.g.,: Iter: 0 MSE: 0.18406609
                #
                # '''if i % 20000 == 0 and i != 0:
                #     if not os.path.exists(model_directory):
                #         os.makedirs(model_directory)
                #     saver.save(sess,model_directory+'/model-'+str(i)+'.ckpt')
                #     print ("Save Model")'''
                #

                #
                # ###################################################################
                # #### compute the mse of validation data ###########################
                # bs_test = test_batch_size
                # batch_index2 = np.random.randint(0, N2, size=bs_test)
                #
                # test_gt_batch = gt2[batch_index2, :, :, :]
                # test_pan_batch = pan2[batch_index2, :, :]
                # test_ms_lr_batch = ms_lr2[batch_index2, :, :, :]
                # test_lms_batch = lms2[batch_index2, :, :, :]
                #
                # pan_hp_batch = get_edge(test_pan_batch)
                # test_pan_hp_batch = pan_hp_batch[:, :, :, np.newaxis]  # expand to N*H*W*1
                #
                # test_ms_hp_batch = get_edge(test_ms_lr_batch)
                #
                # # train_gt, train_lms, train_pan, train_ms = get_batch(train_data, bs = train_batch_size)
                # # test_gt_batch, test_lms_batch, test_pan_batch, test_ms_batch = get_batch(test_data, bs=test_batch_size)
                # test_mse_loss, merged = sess.run([test_mse, test_mse_sum],
                #                                  feed_dict={test_gt: test_gt_batch, test_lms: test_lms_batch,
                #                                             test_ms_hp: test_ms_hp_batch, test_pan_hp: test_pan_hp_batch})
                #
                # mse_valid.append(test_mse_loss)  # record the mse of trainning
                #
                # if i % 1000 == 0 and i != 0:
                #     print("Iter: " + str(i) + " Valid MSE: " + str(test_mse_loss))  # print, e.g.,: Iter: 0 MSE: 0.18406609
            if epoch % 1 == 0 and epoch != 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)
                saver.save(sess, model_directory + '/' + str(epoch) + '/model-' + str(i) + '.ckpt')
                print("Save Model")

        end = datetime.datetime.now()
        print('time cost of DualGCN = ', str(end - start) + 's')

        ## finally write the mse info ##
        file = open('train_mse.txt', 'w')  # write the training error into train_mse.txt
        file.write(str(mse_train))
        file.close()

        file = open('valid_mse.txt', 'w')  # write the valid error into valid_mse.txt
        file.write(str(mse_valid))
        file.close()