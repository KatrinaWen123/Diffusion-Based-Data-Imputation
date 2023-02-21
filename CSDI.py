import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turn off tensorflow warnings
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import pickle
import math
import argparse
import datetime
import json
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
import tensorflow_probability as tfp

from transformer import EncoderLayer
from transformer import Encoder

''' Standalone CSDI imputer. The imputer class is located in the last part of notebook'''

ep = pow(10,-7)

def dict_to_tensor(batch, batch_size = 8):
    tmp = []
    observed_data = batch['observed_data'].numpy()
    observed_mask = batch['observed_mask'].numpy()
    gt_mask = batch['gt_mask'].numpy()
    for _ in range(batch_size - batch['observed_data'].shape[0]):
        observed_data = np.append(observed_data, [observed_data[0]], axis = 0)
        observed_mask = np.append(observed_mask, [observed_mask[0]], axis = 0)
        gt_mask = np.append(gt_mask, [gt_mask[0]], axis = 0)
    tmp.append(observed_data)
    tmp.append(observed_mask)
    tmp.append(gt_mask)
    return tf.convert_to_tensor(tmp)

def tensor_to_dict(t):
    batch = {}
    batch['observed_data'] = t[0,:]
    batch['observed_mask'] = t[1,:]
    batch['gt_mask'] = t[2,:]
    tmp = []
    x = tf.range(t.shape[2])
    for _ in range(t.shape[1]): # batch_size
        tmp.append(x)
    batch['timepoints'] = tf.convert_to_tensor(tmp)
    return batch
    
    
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, milestones, gamma, lr):
    self.milestones = milestones
    self.gamma = gamma
    self.lr = lr

  def __call__(self, step):
    #  return self.initial_learning_rate / (step + 1)
    if step < self.milestones[0]:
        return self.lr
    if step < self.milestones[1]:
        return self.lr * self.gamma
    else:
        return self.lr * self.gamma * self.gamma

def custom_loss(y_true, y_pred):
    # calculate loss, using y_pred
    return y_pred

def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=50, path_save=""):
    output_path = f"{path_save}model.pth"
    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    lr_scheduler = MyLRSchedule(milestones=[p1, p2],gamma=0.1, lr = config["lr"] )
    optimizer = Adam(learning_rate=lr_scheduler)

    best_valid_loss = 1e10
    # model.compile(optimizer = optimizer, loss = custom_loss, run_eagerly=True)
    # model.build(input_shape = [3,8,100,14])
    
    for epoch_no in range(1):
        avg_loss = 0
        with tqdm(train_loader, mininterval=5.0, maxinterval=5.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                with tf.GradientTape() as tape:
                    loss = model(dict_to_tensor(train_batch))
                # print('loss', loss)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
                avg_loss += loss
                it.set_postfix(
                    ordered_dict={"avg_epoch_loss": avg_loss / batch_no,"epoch": epoch_no + 1},refresh=False)
                break
        
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            print('validation')
            avg_loss_valid = 0
            with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, valid_batch in enumerate(it, start=1):
                    valid_batch = dict_to_tensor(valid_batch)
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss
                    it.set_postfix(ordered_dict={"valid_avg_epoch_loss":avg_loss_valid/batch_no,"epoch":epoch_no},refresh=False)
                    break
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print("\n best loss is updated to ",avg_loss_valid/batch_no,"at",epoch_no+1)
        try:
            wandb.log({"loss_valid": avg_loss_valid / batch_no})
        except:
            pass
    
    print('save model')
    # model.save('csdi_model')

def calc_denominator(target, eval_points):
    return tf.math.reduce_sum(tf.math.abs(target * eval_points))

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    flag = (target <= forecast)
    flag = tf.cast(flag, dtype=tf.float32)
    return 2 * tf.math.reduce_sum(tf.math.abs((forecast - target) * eval_points * ((flag) * 1.0 - q)))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(tfp.stats.quantiles(forecast[j: j + 1], quantiles[i], axis=1))
        q_pred = tf.concat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
        
    return CRPS / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, path_save=""):
    print('start evaluation')
    mse_total = 0
    mae_total = 0
    evalpoints_total = 0

    all_target = []
    all_observed_point = []
    all_observed_time = []
    all_evalpoint = []
    all_generated_samples = []
    with tqdm(test_loader, mininterval=5.0, maxinterval=5.0) as it:
        for batch_no, test_batch in enumerate(it, start=1):
            output = model.evaluate(dict_to_tensor(test_batch), nsample)

            samples, c_target, eval_points, observed_points, observed_time = output
            samples = tf.transpose(samples, perm=[0, 1, 3, 2]) # (B,nsample,L,K)
            c_target = tf.transpose(c_target, perm=[0, 2, 1])  # (B,L,K)
            eval_points = tf.transpose(eval_points, perm=[0, 2, 1])
            observed_points = tf.transpose(observed_points, perm=[0, 2, 1])

            samples_median = tf.math.reduce_mean(samples, axis=1)
            all_target.append(c_target)
            all_evalpoint.append(eval_points)
            all_observed_point.append(observed_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(samples)

            mse_current = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
            mae_current = (tf.math.abs((samples_median - c_target) * eval_points)) * scaler

            mse_total += tf.reduce_sum(mse_current)
            mae_total += tf.reduce_sum(mae_current)
            evalpoints_total += tf.reduce_sum(eval_points)
            
            it.set_postfix(ordered_dict={
                    "rmse_total": np.sqrt(mse_total / (evalpoints_total+ep)),
                    "mae_total": mae_total / (evalpoints_total+ep),
                    "batch_no": batch_no}, refresh=True)
            
        with open(f"{path_save}generated_outputs_nsample"+str(nsample)+".pk","wb") as f:
            all_target = tf.concat(all_target, axis=0)
            all_evalpoint = tf.concat(all_evalpoint, axis=0)
            all_observed_point = tf.concat(all_observed_point, axis=0)
            all_observed_time = tf.concat(all_observed_time, axis=0)
            all_generated_samples = tf.concat(all_generated_samples, axis=0)

            pickle.dump(
                [
                    all_generated_samples,
                    all_target,
                    all_evalpoint,
                    all_observed_point,
                    all_observed_time,
                    scaler,
                    mean_scaler,
                ],
                f,
            )

        CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

        with open(f"{path_save}result_nsample" + str(nsample) + ".pk", "wb") as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / (evalpoints_total+ep)),
                        mae_total / (evalpoints_total+ep), 
                        CRPS
                    ], 
                    f)
                print("RMSE:", np.sqrt(mse_total / (evalpoints_total+ep)))
                print("MAE:", mae_total / (evalpoints_total+ep))
                print("CRPS:", CRPS)
    return all_generated_samples.numpy()


def get_dataloader_train_impute(series,
                                batch_size=4,
                                missing_ratio_or_k=0.1,
                                train_split=0.7,
                                valid_split=0.9,
                                len_dataset=10,
                                masking='rm',
                               path_save='',
                               ms=None):
    indlist = np.arange(len_dataset)

    tr_i, v_i, te_i = np.split(indlist,
                               [int(len(indlist) * train_split),
                                int(len(indlist) * (train_split + valid_split))])

    
    train_dataset = Custom_Train_Dataset(series=series, use_index_list=tr_i,
                                         missing_ratio_or_k=missing_ratio_or_k, 
                                         masking=masking, path_save=path_save, ms=1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # print(train_loader.__len__())
    # print(train_loader.__getitem__(0))

    valid_dataset = Custom_Train_Dataset(series=series, use_index_list=v_i, 
                                         missing_ratio_or_k=missing_ratio_or_k, 
                                         masking=masking, path_save=path_save)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Custom_Train_Dataset(series=series, use_index_list=te_i, 
                                        missing_ratio_or_k=missing_ratio_or_k, 
                                        masking=masking, path_save=path_save)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader

def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)

    masks = tf.identity(tf.reshape(observed_masks,-1))
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    masks = np.array(masks)
    masks[miss_indices] = False
    gt_masks = tf.reshape(masks, observed_masks.shape)
    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = np.array(gt_masks).astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_nrm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)

    for channel in range(gt_masks.shape[1]):
        s_nan = random.choice(list_of_segments_index)
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks


def mask_missing_train_bm(data, k_segments=5):
    observed_values = np.array(data)
    observed_masks = ~np.isnan(observed_values)
    gt_masks = observed_masks.copy()
    length_index = np.array(range(data.shape[0]))
    list_of_segments_index = np.array_split(length_index, k_segments)
    s_nan = random.choice(list_of_segments_index)

    for channel in range(gt_masks.shape[1]):
        gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.x = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        # print('len(self.x)',len(self.x))
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # batch_x = self.x[idx * self.batch_size:(idx + 1) *self.batch_size]
        batch_x = {}
        observed_data = []
        observed_mask = []
        gt_mask = []
        timepoints = []
        for i in range(idx * self.batch_size,np.minimum((idx + 1) *self.batch_size, len(self.x))):
            data = self.x.__getitem__(i)
            observed_data.append(data['observed_data'])
            observed_mask.append(data['observed_mask'])
            gt_mask.append(data['gt_mask'])
            timepoints.append(data['timepoints'])
        batch_x['observed_data'] = tf.convert_to_tensor(observed_data)
        batch_x['observed_mask'] = tf.convert_to_tensor(observed_mask)
        batch_x['gt_mask'] = tf.convert_to_tensor(gt_mask)
        batch_x['timepoints'] = tf.convert_to_tensor(timepoints)
        return batch_x
        
class Custom_Train_Dataset:
    def __init__(self, series, path_save='', use_index_list=None, missing_ratio_or_k=0.0, masking='rm', ms=None):
        self.series = series
        self.length = series.shape[1]
        self.n_channels = series.shape[2]

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        
        path = ''
        if not os.path.isfile(path):  # if datasetfile is none, create
            for sample in series:
                sample = tf.convert_to_tensor(sample)
                sample = tf.stop_gradient(sample).numpy()
                if masking == 'rm':
                    observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, missing_ratio_or_k)
                    observed_values = tf.convert_to_tensor(observed_values)
                    observed_masks = tf.convert_to_tensor(observed_masks)
                    gt_masks = tf.convert_to_tensor(gt_masks)
                elif masking == 'nrm':
                    observed_values, observed_masks, gt_masks = mask_missing_train_nrm(sample, missing_ratio_or_k)
                    observed_values = tf.convert_to_tensor(observed_values)
                    observed_masks = tf.convert_to_tensor(observed_masks)
                    gt_masks = tf.convert_to_tensor(gt_masks)
                elif masking == 'bm':
                    observed_values, observed_masks, gt_masks = mask_missing_train_bm(sample, missing_ratio_or_k)
                    observed_values = tf.convert_to_tensor(observed_values)
                    observed_masks = tf.convert_to_tensor(observed_masks)
                    gt_masks = tf.convert_to_tensor(gt_masks)
                    
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                
        
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def Conv1d_with_init(in_channels, out_channels, kernel_size, initializer = None):
    if initializer == None:
        layer = tf.keras.layers.Conv1D(out_channels, kernel_size, kernel_initializer='HeNormal')
    else:
        layer = tf.keras.layers.Conv1D(out_channels, kernel_size, kernel_initializer=initializer)
    return layer

def get_torch_trans(heads=8, layers=1, channels=64):
    return Encoder(num_layers=layers, d_model = channels, num_heads = heads, dff = 64)


class DiffusionEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.embedding = self._build_embedding(num_steps, embedding_dim / 2)
        # self.register_buffer(
        #     "embedding",
        #     self._build_embedding(num_steps, embedding_dim / 2),
        #     persistent=False)
        self.projection1 = Dense(projection_dim)
        self.projection2 = Dense(projection_dim)

    def call(self, diffusion_step):
        x = []
        for i in range(diffusion_step.shape[0]):
            x.append(self.embedding[diffusion_step[i]])
        x = tf.stack(x)
        x = self.projection1(x)
        x = tf.nn.silu(x)
        x = self.projection2(x)
        x = tf.nn.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = tf.expand_dims(tf.range(num_steps), 1) # (T,1)
        steps = tf.cast(steps, dtype = tf.float32)
        frequencies = 10.0 ** (tf.range(dim) / (dim - 1) * 4.0)  
        frequencies = tf.expand_dims(frequencies, 0) # (1,dim)
        table = tf.matmul(steps, frequencies)  # (T,dim)
        table = tf.concat([tf.math.sin(table), tf.math.cos(table)], axis=1)  # (T,dim*2)
        return table

    
class diff_CSDI(tf.keras.layers.Layer):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.layers = config["layers"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"])

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1, initializer = 'Zeros')

        self.residual_layers1 = ResidualBlock(side_dim=config["side_dim"],channels=self.channels,diffusion_embedding_dim=config["diffusion_embedding_dim"],nheads=config["nheads"])
        self.residual_layers2 = ResidualBlock(side_dim=config["side_dim"],channels=self.channels,diffusion_embedding_dim=config["diffusion_embedding_dim"],nheads=config["nheads"])
        self.residual_layers3 = ResidualBlock(side_dim=config["side_dim"],channels=self.channels,diffusion_embedding_dim=config["diffusion_embedding_dim"],nheads=config["nheads"])
        self.residual_layers4 = ResidualBlock(side_dim=config["side_dim"],channels=self.channels,diffusion_embedding_dim=config["diffusion_embedding_dim"],nheads=config["nheads"])
        
        # self.residual_layers = ResidualBlock(
        #             side_dim=config["side_dim"],
        #             channels=self.channels,
        #             diffusion_embedding_dim=config["diffusion_embedding_dim"],
        #             nheads=config["nheads"],
        #         )

    def call(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = tf.reshape(x,[B, inputdim, K * L])
        x = tf.transpose(x, perm = [0,2,1])
        x = self.input_projection(x)
        x = tf.transpose(x, perm = [0,2,1])
        x = tf.nn.relu(x)
        x = tf.reshape(x, [B, self.channels, K, L])
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        x, skip_connection = self.residual_layers1(x, cond_info, diffusion_emb)
        skip.append(skip_connection)
        x, skip_connection = self.residual_layers2(x, cond_info, diffusion_emb)
        skip.append(skip_connection)
        x, skip_connection = self.residual_layers3(x, cond_info, diffusion_emb)
        skip.append(skip_connection)
        x, skip_connection = self.residual_layers4(x, cond_info, diffusion_emb)
        skip.append(skip_connection)
        # for layer in self.residual_layers.layers:
        #     x, skip_connection = layer(x, cond_info, diffusion_emb)
        #     skip.append(skip_connection)
        x = tf.reduce_sum(tf.stack(skip), axis=0) / math.sqrt(4)
        x = tf.reshape(x, [B, self.channels, K * L])
        x = tf.transpose(x, perm=[0,2,1])
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = tf.transpose(x, perm=[0,2,1])
        x = tf.nn.relu(x)
        x = tf.transpose(x, perm=[0,2,1])
        x = self.output_projection2(x)  # (B,1,K*L)
        x = tf.transpose(x, perm=[0,2,1])
        x = tf.reshape(x, [B, K, L])
        return x

    
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = Dense(channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]), perm=[0, 2, 1, 3]), [B * K, channel, L])
        y = tf.transpose(self.time_layer(tf.transpose(y, perm=[2, 0, 1])), perm=[1,2,0])
        y = tf.reshape(tf.transpose(tf.reshape(y, [B, K, channel, L]), perm=[0, 2, 1,3]), [B, channel, K * L])
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = tf.reshape(tf.transpose(tf.reshape(y,[B, channel, K, L]), perm=[0, 3, 1, 2]), [B * L, channel, K])
        y = tf.transpose(self.feature_layer(tf.transpose(y, perm=[2, 0, 1])), perm=[1,2,0])
        y = tf.reshape(tf.transpose(tf.reshape(y, [B, L, channel, K]), perm=[0, 2, 3, 1]), [B, channel, K * L])
        return y

    def call(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = tf.reshape(x, [B, channel, K * L])

        # diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        diffusion_emb = tf.expand_dims(self.diffusion_projection(diffusion_emb), -1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = tf.transpose(y, perm=[0,2,1])
        y = self.mid_projection(y)  # (B,2*channel,K*L)
        y = tf.transpose(y, perm=[0,2,1])
        
        _, cond_dim, _, _ = cond_info.shape
        cond_info = tf.reshape(cond_info, [B, cond_dim, K * L])
        cond_info = tf.transpose(cond_info, perm=[0,2,1])
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        cond_info = tf.transpose(cond_info, perm=[0,2,1])
        y = y + cond_info

        
        gate, filter = tf.split(y, 2, axis=1)
        y = tf.math.sigmoid(gate) * tf.math.tanh(filter)  # (B,channel,K*L)
        y = tf.transpose(y, perm=[0,2,1])
        y = self.output_projection(y)
        y = tf.transpose(y, perm=[0,2,1])

        residual, skip = tf.split(y, 2, axis=1)
        x = tf.reshape(x, base_shape)
        residual = tf.reshape(residual, base_shape)
        skip = tf.reshape(skip, base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CSDI_Custom(tf.keras.Model):
    def __init__(self, config, batch_size =8, target_dim=35):
        super(CSDI_Custom, self).__init__()
        self.target_dim = target_dim
        self.batch_size = batch_size
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.shape0 = config["model"]["shape0"]
        self.shape1 = config["model"]["shape1"]
        self.shape2 = config["model"]["shape2"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = tf.keras.layers.Embedding(input_dim=self.target_dim, output_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(config_diff["beta_start"], config_diff["beta_end"], self.num_steps)

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(self.alpha,dtype = float), axis = 1), 1)

    def process_data(self, batch):
        observed_data = batch["observed_data"]
        observed_mask = batch["observed_mask"]
        observed_tp = batch["timepoints"]
        gt_mask = batch["gt_mask"]

        observed_data = tf.transpose(observed_data, perm=[0, 2, 1])
        observed_mask = tf.transpose(observed_mask, perm=[0, 2, 1])
        gt_mask = tf.transpose(gt_mask, perm=[0, 2, 1])

        # ? wt: cut_length usuage
        cut_length = tf.zeros(self.batch_size)
        for_pattern_mask = observed_mask

        return (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,cut_length)
 
 
    def time_embedding(self, pos, d_model=128):
        # pe = np.zeros([pos.shape[0], pos.shape[1], d_model])
        pe = []
        pe_single = []
        position = tf.expand_dims(pos, 2)
        position = tf.cast(position, dtype=tf.float32)
        div_term = 1.0 / tf.math.pow(10000.0, tf.range(0.0, d_model, 2)/ d_model)
        div_term = tf.cast(div_term, dtype=tf.float32)
        sin_value = tf.math.sin(position * div_term)[0]
        cos_value = tf.math.cos(position * div_term)[0]
        sin_value = tf.transpose(sin_value, perm=[1,0])
        cos_value = tf.transpose(cos_value, perm=[1,0])
        for i in range(sin_value.shape[0]):
            pe_single.append(sin_value[i])
            pe_single.append(cos_value[i])
        pe_single = tf.stack(pe_single)
        pe_single = tf.transpose(pe_single, perm=[1,0])
        for i in range(pos.shape[0]):
            pe.append(pe_single)
        pe = tf.stack(pe)
        return pe

    # generate randmask
    def get_randmask(self, observed_mask):
        rand_for_mask = tf.random.uniform(shape=[self.shape0, self.shape2, self.shape1], minval=-1, maxval=1, dtype=tf.int32) 
        rand_for_mask = tf.cast(rand_for_mask, dtype = tf.float32) * observed_mask
        # for i in range(observed_mask.shape[0]):
            # sample_ratio = np.random.rand()  
            # num_observed = tf.math.reduce_sum(observed_mask[i])
            # num_masked = tf.math.round(num_observed * sample_ratio)
            # num_masked = tf.cast(num_masked, dtype=tf.int32)
            # rand_for_mask[i][tf.math.top_k(rand_for_mask[i], k = num_masked).indices] = -1
        cond_mask = tf.reshape((rand_for_mask > 0),observed_mask.shape)
        return cond_mask

    
    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(cond_mask.shape[0]):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else: 
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    
    
    def get_side_info(self, observed_tp, cond_mask):
        
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = tf.repeat(tf.expand_dims(time_embed,2), K, axis=2) 
        feature_embed = self.embed_layer(tf.range(self.target_dim))  # (K,emb)
        feature_embed = tf.expand_dims(tf.expand_dims(feature_embed, 0),0)
        feature_embed = tf.repeat(tf.repeat(feature_embed,B,axis=0), L, axis = 1)
        # feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = tf.concat([time_embed, feature_embed], -1)  # (B,L,K,*)
        side_info = tf.transpose(side_info, perm=[0, 3, 2, 1]) # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = tf.expand_dims(cond_mask, 1) # (B,1,K,L)
            side_mask = tf.cast(side_mask, dtype=tf.float32)
            side_info = tf.concat([side_info, side_mask], 1)

        return side_info

    
    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, side_info, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t)
            loss_sum += loss
            
        return loss_sum / self.num_steps

    
    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (tf.ones(B) * set_t)
            t = tf.cast(t, dtype=tf.int32)
        else:
            # t = torch.randint(0, self.num_steps, [B]).to(self.device)
            t = tf.random.uniform([B], minval=0, maxval=self.num_steps, dtype=tf.int32)
        current_alpha = []
        for i in range(B):
            current_alpha.append(self.alpha_torch[t[i]])
        current_alpha = tf.stack(current_alpha) # (B,1,1)
        # current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = tf.random.normal(shape = observed_data.shape)
        current_alpha = tf.cast(current_alpha, dtype=tf.float32)
        noisy_data = (tf.math.pow(current_alpha, 0.5)) * observed_data + tf.math.pow((1.0 - current_alpha), 0.5) * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        observed_mask = tf.cast(observed_mask, dtype=tf.float32)
        cond_mask = tf.cast(cond_mask, dtype=tf.float32)
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        
        num_eval = tf.math.reduce_sum(target_mask)
        # print('residual',residual.shape)
        # print('tf.math.reduce_sum(residual ** 2)', tf.math.reduce_sum(residual ** 2))
        # if num_eval <= 0:
        #     num_eval = 1
        num_eval = tf.math.maximum(num_eval, tf.convert_to_tensor(1.0))
        loss = tf.math.reduce_sum(residual ** 2) / num_eval
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_mask = tf.cast(cond_mask, dtype=tf.float32)
        if self.is_unconditional == True:
            total_input = tf.expand_dims(noisy_data, 1)  # (B,1,K,L)
        else:
            cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
            noisy_target = tf.expand_dims(((1 - cond_mask) * noisy_data), 1)
            total_input = tf.concat([cond_obs, noisy_target], 1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        # imputed_samples = tf.zeros([B, n_samples, K, L])

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = tf.random.normal(shape=noisy_obs.shape)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = tf.random.normal(shape=observed_data.shape)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = tf.expand_dims(diff_input, 1)  # (B,1,K,L)
                else:
                    cond_obs = tf.expand_dims(cond_mask * observed_data, 1)
                    noisy_target = tf.expand_dims((1 - cond_mask) * current_sample, 1)
                    diff_input = tf.concat([cond_obs, noisy_target], axis=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, tf.convert_to_tensor([t]))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = tf.random.normal(shape=current_sample.shape)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise
            
            if i == 0:
                imputed_samples = tf.expand_dims(current_sample, axis = 1)
            else:
                imputed_samples = tf.concat([imputed_samples, tf.expand_dims(current_sample, axis = 1)], axis = 1)
            # imputed_samples[:, i] = current_sample
            
        return imputed_samples

    
    def call(self,batch, is_train=1):
        # batch = {}
        # batch['observed_data'] = observed_data
        # batch['observed_mask'] = observed_mask
        # batch['gt_mast'] = gt_mast
        # batch['timepoints'] = timepoints
        batch = tensor_to_dict(batch)

        (observed_data,observed_mask,observed_tp,gt_mask,for_pattern_mask,_) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss = loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)
        print('final result', loss)
        return loss

    def evaluate(self, batch, n_samples):
        batch = tensor_to_dict(batch)
        (observed_data,observed_mask,observed_tp,gt_mask,_,cut_length) = self.process_data(batch)
        cond_mask = gt_mask
        target_mask = observed_mask - cond_mask
        side_info = self.get_side_info(observed_tp, cond_mask)
        samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        target_mask_res = np.ones(shape = target_mask.shape)
        cut_length = tf.cast(cut_length, dtype = tf.int32)
        for i in range(target_mask_res.shape[0]):  
            target_mask_res[i, ..., 0: cut_length[i]] = 0
        target_mask_res = tf.convert_to_tensor(target_mask_res) 
        target_mask_res = tf.cast(target_mask_res, dtype = tf.float32) 
        target_mask = tf.cast(target_mask, dtype = tf.float32)
        target_mask_res = target_mask_res * target_mask
        return samples, observed_data, target_mask, observed_mask, observed_tp



class Custom_Impute_Dataset:
    def __init__(self, series, mask, use_index_list=None, path_save=''):
        self.series = series
        self.n_channels = series.shape[2]
        self.length = series.shape[1]
        self.mask = mask 

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        path = ''
        if not os.path.isfile(path):  # if datasetfile is none, create
            for sample in series:
                sample = tf.convert_to_tensor(sample)
                sample = tf.stop_gradient(sample).numpy()
                observed_masks = sample.copy()
                observed_masks[observed_masks!=0] = 1 
                gt_masks = mask
                
                #observed_values, observed_masks, gt_masks = mask_missing_impute(sample, mask)
                
                self.observed_values.append(sample)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)

                
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
 

def get_dataloader_impute(series, mask, batch_size=4, len_dataset=100):
    indlist = np.arange(len_dataset)
    impute_dataset = Custom_Impute_Dataset(series=series, use_index_list=indlist,mask=mask)
    impute_loader = DataLoader(impute_dataset, batch_size=batch_size, shuffle=False)
    return impute_loader
 
class CSDIImputer(tf.keras.layers.Layer):
    def __init__(self):
        np.random.seed(0)
        random.seed(0)
        self.config = {}
        
        '''
        CSDI imputer
        3 main functions:
        a) training based on random missing, non-random missing, and blackout masking.
        b) loading weights of already trained model
        c) impute samples in inference. Note, you must manually load weights after training for inference.
        '''

    def train(self,
              series,
              masking ='rm',
              missing_ratio_or_k = 0.0,
              train_split = 0.7,
              valid_split = 0.2,
              epochs = 2,
              samples_generate = 10,
              path_save = "",
              batch_size = 16,
              lr = 1.0e-3,
              layers = 4,
              channels = 64,
              nheads = 8,
              difussion_embedding_dim = 128,
              beta_start = 0.0001,
              beta_end = 0.5,
              num_steps = 50,
              schedule = 'quad',
              is_unconditional = 0,
              timeemb = 128,
              featureemb = 16,
              target_strategy = 'random',
             ):
        
        '''
        CSDI training function. 
       
       
        Requiered parameters
        -series: Assumes series of shape (Samples, Length, Channels).
        -masking: 'rm': random missing, 'nrm': non-random missing, 'bm': black-out missing.
        -missing_ratio_or_k: missing ratio 0 to 1 for 'rm' masking and k segments for 'nrm' and 'bm'.
        -path_save: full path where to save model weights, configuration file, and means and std devs for de-standardization in inference.
        
        Default parameters
        -train_split: 0 to 1 representing the percentage of train set from whole data.
        -valid_split: 0 to 1. Is an adition to train split where 1 - train_split - valid_split = test_split (implicit in method).
        -epochs: number of epochs to train.
        -samples_generate: number of samples to be generated.
        -batch_size: batch size in training.
        -lr: learning rate.
        -layers: difussion layers.
        -channels: number of difussion channels.
        -nheads: number of difussion 'heads'.
        -difussion_embedding_dim: difussion embedding dimmensions. 
        -beta_start: start noise rate.
        -beta_end: end noise rate.
        -num_steps: number of steps.
        -schedule: scheduler. 
        -is_unconditional: conditional or un-conditional imputation. Boolean.
        -timeemb: temporal embedding dimmensions.
        -featureemb: feature embedding dimmensions.
        -target_strategy: strategy of masking. 
        -wandbiases_project: weight and biases project.
        -wandbiases_experiment: weight and biases experiment or run.
        -wandbiases_entity: weight and biases entity. 
        '''
       
        config = {}
        
        config['train'] = {}
        config['train']['epochs'] = epochs
        config['train']['batch_size'] = batch_size
        config['train']['lr'] = lr
        config['train']['train_split'] = train_split
        config['train']['valid_split'] = valid_split
        config['train']['path_save'] = path_save
        
       
        config['diffusion'] = {}
        config['diffusion']['layers'] = layers
        config['diffusion']['channels'] = channels
        config['diffusion']['nheads'] = nheads
        config['diffusion']['diffusion_embedding_dim'] = difussion_embedding_dim
        config['diffusion']['beta_start'] = beta_start
        config['diffusion']['beta_end'] = beta_end
        config['diffusion']['num_steps'] = num_steps
        config['diffusion']['schedule'] = schedule
        
        config['model'] = {} 
        config['model']['missing_ratio_or_k'] = missing_ratio_or_k
        config['model']['is_unconditional'] = is_unconditional
        config['model']['timeemb'] = timeemb
        config['model']['featureemb'] = featureemb
        config['model']['target_strategy'] = target_strategy
        config['model']['masking'] = masking
        
        config['model']['shape0'] = 8
        config['model']['shape1'] = 100
        config['model']['shape2'] = 14
        self.config = config
        
        # print(json.dumps(config, indent=4))

        # config_filename = path_save + "config_csdi_training"
        # # print('configuration file name:', config_filename)
        # with open(config_filename + ".json", "w") as f:
        #     json.dump(config, f, indent=4)
        


        train_loader, valid_loader, test_loader = get_dataloader_train_impute(
            series=series,
            train_split=config["train"]["train_split"],
            valid_split=config["train"]["valid_split"],
            len_dataset=series.shape[0],
            batch_size=config["train"]["batch_size"],
            missing_ratio_or_k=config["model"]["missing_ratio_or_k"],
            masking=config['model']['masking'],
            path_save=config['train']['path_save'])
        
        batch_size = 8
        model = CSDI_Custom(config, target_dim=series.shape[2])

        train(model=model,
              config=config["train"],
              train_loader=train_loader,
              valid_loader=valid_loader,
              path_save=config['train']['path_save'])

        # evaluate(model=model,
        #          test_loader=test_loader,
        #          nsample=samples_generate,
        #          scaler=1,
        #          path_save=config['train']['path_save'])
        return model
        
        
    def load_weights(self, 
                     path_load_model='',
                     path_config=''):
        
        self.path_load_model_dic = path_load_model
        self.path_config = path_config
    
    
        '''
        Load weights and configuration file for inference.
        
        path_load_model: load model weights
        path_config: load configuration file
        '''
    

    def impute(self,
               model,
               sample,
               mask,
               n_samples = 2,
               ):
        
        '''
        Imputation function 
        sample: sample(s) to be imputed (Samples, Length, Channel)
        mask: mask where values to be imputed. 0's to impute, 1's to remain. 
        n_samples: number of samples to be generated
        return imputations with shape (Samples, N imputed samples, Length, Channel)
        '''
        
        if len(sample.shape) == 2:
            self.series_impute = tf.convert_to_tensor(np.expand_dims(sample, axis=0))
        elif len(sample.shape) == 3:
            self.series_impute = sample

        # with open(self.path_config, "r") as f:
        #     config = json.load(f)

        test_loader = get_dataloader_impute(series=self.series_impute,len_dataset=len(self.series_impute),
                                            mask=mask, batch_size=self.config['train']['batch_size'])

        # model = CSDI_Custom(config, self.device, target_dim=self.series_impute.shape[2]).to(self.device)
        # model = keras.models.load_model(self.path_load_model_dic)

        imputations = evaluate(model=model,
                                test_loader=test_loader,
                                nsample=n_samples,
                                scaler=1,
                                path_save='')
        
        # indx_imputation = ~mask.astype(bool)
            
        # original_sample_replaced =[]
        
        # sample = tf.convert_to_tensor(sample)
        # for original_sample, single_n_samples in zip(sample, imputations): # [x,x,x] -> [x,x] & [x,x,x,x] -> [x,x,x]            
        #     single_sample_replaced = []
        #     for sample_generated in single_n_samples:  # [x,x] & [x,x,x] -> [x,x]
        #         sample_out = original_sample.copy()                         
        #         sample_out[indx_imputation] = sample_generated[indx_imputation]
        #         single_sample_replaced.append(sample_out)
        #     original_sample_replaced.append(single_sample_replaced)
            
        # output = np.array(original_sample_replaced)
        
        print('finished all')
        return


imputer = CSDIImputer()
train_path = 'train_mujoco.npy'
data_train = np.load(train_path)
data_train = data_train[:50]
test_path = 'test_mujoco.npy'
data_test = np.load(test_path)
data_test = data_test[:20]

model = imputer.train(data_train, masking='rm',path_save='' , batch_size = 8) # for training
mask = np.random.randint(0*data_test[0], 2)
imputations = imputer.impute(model, data_test, mask) # sampling