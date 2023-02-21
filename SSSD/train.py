import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turn off tensorflow warnings
import tensorflow as tf
import argparse
import json
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from util import training_loss, calc_diffusion_hyperparams
from util import get_mask_mnr, get_mask_bm, get_mask_rm

# from imputers.DiffWaveImputer import DiffWaveImputer
# from imputers.SSSDSAImputer import SSSDSAImputer
from SSSDS4Imputer import SSSDS4Imputer


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])
    
    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key]

    net = SSSDS4Imputer(**model_config)

    # define optimizer
    optimizer = Adam(net.parameters(), lr=learning_rate)
        
    
    ### Custom data loading and reshaping ###
        
    training_data = np.load(trainset_config['train_data_path'])
    training_data = np.split(training_data, 160, 0)
    training_data = np.array(training_data)
    training_data = tf.cast(tf.convert_to_tensor(training_data), dtype=tf.float32)
    print('Data loaded')

    
    
    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch in training_data:

            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k)
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)

            mask = tf.transpose(transposed_mask, perm=[1,0])
            mask = tf.repeat(mask, repeats=[batch.shape[0], 1,1])
            mask = tf.cast(mask, dtype=tf.float32)
            loss_mask = ~tf.cast(mask, dtype = tf.bool)
            batch = tf.transpose(batch, perm=[0, 2, 1])

            assert batch.shape == mask.shape == loss_mask.shape

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            with tf.GradientTape() as tape:
                loss = training_loss(net, tf.keras.losses.MeanSquaredError(), X, diffusion_hyperparams,
                                 only_generate_missing=only_generate_missing)

            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, net.trainable_variables))

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss))

            # save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                net.save('SSSD_model')
                print('model at iteration %s is saved' % n_iter)
            
            print('n_iter',n_iter)
            n_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/config_SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train(**train_config)
