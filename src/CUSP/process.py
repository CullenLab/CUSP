import time
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.cluster import DBSCAN
import torch
from tqdm import tqdm
import umap

# local import
from CUSP.model import *


class CUSP_pipeline():
    def __init__(self,params):
        self.fs = params['fs']
        self.fs_ms = params['fs_ms']
        self.N_ch = params['N_ch']
        
        self.L_segment_batch = params['L_segment_batch']
        self.L_segment_len = params['L_segment_len']
        self.L_segment_step = params['L_segment_step']

        self.CS_seg_ms = params['CS_seg_ms']
        self.L_segment_edge_discard = params['L_segment_edge_discard']

        self.DBSCAN_eps = params['DBSCAN_eps']
        self.CS_peak_sign = params['CS_peak_sign']
        self.CS_detect_threshold_quantile = params['CS_detect_threshold_quantile']

        # self.channel_range = params['channel_range']
        self.lfp_filt_order = params['lfp_filt_order']
        self.lfp_filt_band = params['lfp_filt_band']
        self.ap_filt_order = params['ap_filt_order']
        self.ap_filt_band = params['ap_filt_band']

        self.ch_anchor = params['ch_anchor']

        self.model_path = params['model_path']

        self.data_path = params['data_path']

        self.if_cluster_waveform = params['if_cluster_waveform']

        self.enable_plot = params['enable_plot']

    def load_model(self):
        print('Load model: '+self.model_path)
        torch.cuda.empty_cache()
        self.device = torch.device("cuda")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.load(self.model_path,weights_only=False)
        
        self.model.to(self.device)
        self.model.eval()

    def load_data(self):
        if self.data_path.endswith('.mat'):
            self.load_data_from_mat()
        elif self.data_path.endswith('.bin'):
            self.load_data_from_bin()
        else:
            raise ValueError("Unsupported file format. Please provide a .mat or .bin file.")

    def load_data_from_mat(self):
        print('Load _Cal.mat data: '+self.data_path)
        with h5py.File(self.data_path, 'r') as f:
            self.Neural = f['Data']['Neural'][:].astype(np.float32).T # time*channel

        self.L_data = np.size(self.Neural,0)
        print("Data length: "+str(self.L_data)+", Data total time: "+str(self.L_data/self.fs)+" second")

    def load_data_from_bin(self):
        print('Load .bin data: '+self.data_path)
        with open(self.data_path, 'rb') as f:
            self.Neural = np.fromfile(f, dtype=np.int16).astype(np.float32).reshape(-1,self.N_ch) # time*channel

        self.L_data = np.size(self.Neural,0)
        print("Data length: "+str(self.L_data)+", Data total time: "+str(self.L_data/self.fs)+" second")

    def preprocess(self):

        print("Determine channel range based on ch_anchor")
        channel_span_start = 4*np.floor(self.ch_anchor/4)-4
        channel_span_end = 4*np.floor(self.ch_anchor/4)+8
        if channel_span_start < 0:
            channel_span_start = 0
            channel_span_end = 12
        elif channel_span_end > self.N_ch:
            channel_span_start = self.N_ch-12
            channel_span_end = self.N_ch
        self.channel_span = np.arange(channel_span_start,channel_span_end).astype(int) # 12 channels near ch_anchor

        print("De-mean per channel")
        self.Neural = self.Neural - np.mean(self.Neural,axis=0)
        
        print("De-median across channel")
        self.Neural = self.Neural - np.median(self.Neural,axis=1)[:,np.newaxis]

        print("De-median local channel")
        Neural_copy = self.Neural.copy()
        x = np.zeros((np.size(Neural_copy,0),len(self.channel_span)))
        for i in range(len(self.channel_span)):
            local_channel_demedian = np.arange(np.max([0,self.channel_span[i]-16]),np.min([self.channel_span[i]+16,self.N_ch])).astype(int)
            x[:,i] = Neural_copy[:,self.channel_span[i]] - np.median(Neural_copy[:,local_channel_demedian],axis=1)
        del Neural_copy

        print("Filt LFP")
        sos_lfp = butter(self.lfp_filt_order,self.lfp_filt_band,'bandpass',fs=self.fs,output='sos')
        self.lfp = sosfiltfilt(sos_lfp,x,axis=0).T # channel*time

        print("Filt AP")
        sos_ap = butter(self.ap_filt_order,self.ap_filt_band,'bandpass',fs=self.fs,output='sos')
        self.ap = sosfiltfilt(sos_ap,x,axis=0).T # channel*time

    def model_predict(self):
        print("Model predict")
        model_out_final = -torch.ones((self.L_data,))
        start_time = time.time()
        # batch process
        for tstart in tqdm(range(0,self.L_data,self.L_segment_step*self.L_segment_batch)):
            # if not at end and can fulfil full batch
            if (tstart+self.L_segment_step*(self.L_segment_batch-1)+self.L_segment_len) <= self.L_data:
                temp_ap = [torch.from_numpy(self.ap[:,(tstart+self.L_segment_step*i_batch):(tstart+self.L_segment_step*i_batch+self.L_segment_len)][np.newaxis,:,:].astype(np.float32).copy()) for i_batch in range(self.L_segment_batch)]
                temp_ap = torch.cat(temp_ap,dim=0)
                temp_lfp = [torch.from_numpy(self.lfp[:,(tstart+self.L_segment_step*i_batch):(tstart+self.L_segment_step*i_batch+self.L_segment_len)][np.newaxis,:,:].astype(np.float32).copy()) for i_batch in range(self.L_segment_batch)]
                temp_lfp = torch.cat(temp_lfp,dim=0)

                model_out = torch.sigmoid(self.model((temp_lfp[:,None,:,:]).to(self.device),temp_ap[:,None,:,:].to(self.device)))

                del temp_ap,temp_lfp
                torch.cuda.empty_cache()

                # ignore_model_out edge
                model_out[:,:self.L_segment_edge_discard] = -1
                model_out[:,-self.L_segment_edge_discard:] = -1

                for i_batch in range(self.L_segment_batch):
                    model_out_final[(tstart+self.L_segment_step*i_batch):(tstart+self.L_segment_step*i_batch+self.L_segment_len)] = \
                                                            torch.tensor(model_out.cpu().detach().numpy())[i_batch,:] \
                                                            * (torch.tensor(model_out.cpu().detach().numpy())[i_batch,:] != -1) \
                                                            * (model_out_final[(tstart+self.L_segment_step*i_batch):(tstart+self.L_segment_step*i_batch+self.L_segment_len)] != -1)
                del model_out
                torch.cuda.empty_cache()

            else: # at end and cannot fulfil full batch
                # then do it per 1 batch
                for t in range(tstart,self.L_data,self.L_segment_step):
                    # if full segment length
                    if t + self.L_segment_len <= self.L_data:
                        temp_ap = torch.tensor(self.ap[:,(t):(t+self.L_segment_len)][np.newaxis,:,:].astype(np.float32).copy())
                        temp_lfp = torch.tensor(self.lfp[:,(t):(t+self.L_segment_len)][np.newaxis,:,:].astype(np.float32).copy())

                        model_out = torch.sigmoid(self.model(temp_lfp[:,None,:,:].to(self.device),temp_ap[:,None,:,:].to(self.device)))

                        del temp_ap,temp_lfp
                        torch.cuda.empty_cache()

                        # ignore_model_out edge
                        model_out[:,:self.L_segment_edge_discard] = -1
                        model_out[:,-self.L_segment_edge_discard:] = -1

                        model_out_final[(t):(t+self.L_segment_len)] = \
                                        torch.tensor(model_out.cpu().detach().numpy())[0,:] \
                                        * (torch.tensor(model_out.cpu().detach().numpy())[0,:] != -1) \
                                        * (model_out_final[(t):(t+self.L_segment_len)] != -1)
                        del model_out
                        torch.cuda.empty_cache()

                    else: # if not full length
                        temp_ap = torch.tensor(self.ap[:,(t):][np.newaxis,:,:].astype(np.float32).copy())
                        temp_lfp = torch.tensor(self.lfp[:,(t):][np.newaxis,:,:].astype(np.float32).copy())

                        model_out = torch.sigmoid(self.model(temp_lfp[:,None,:,:].to(self.device),temp_ap[:,None,:,:].to(self.device)))

                        del temp_ap,temp_lfp
                        torch.cuda.empty_cache()

                        # ignore_model_out edge
                        model_out[:,:self.L_segment_edge_discard] = -1
                        model_out[:,-self.L_segment_edge_discard:] = -1

                        model_out_final[(t):] = \
                                        torch.tensor(model_out.cpu().detach().numpy())[0,:] \
                                        * (torch.tensor(model_out.cpu().detach().numpy())[0,:] != -1) \
                                        * (model_out_final[(t):] != -1)
                        
                        del model_out
                        torch.cuda.empty_cache()
                
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Model predict elapsed time: {elapsed_time} seconds")

        self.model_out_final = model_out_final

    def model_out_conv(self):
        print("Model out conv")
        conv_kernel = np.kaiser(int(5*self.fs_ms), beta=14)
        self.model_out_final_conv = torch.tensor(np.convolve(self.model_out_final.numpy(),conv_kernel,mode='same'))

    def detect_CS_idx(self):
        print("First detect CS, threshold quantile: "+str(self.CS_detect_threshold_quantile))
        cs_detect_x_idx = self.model_out_final_conv > np.quantile(self.model_out_final_conv,self.CS_detect_threshold_quantile)
        # cs_detect_x_idx = np.where(cs_detect_x_idx)[0]
        cs_detect_x_idx = find_peaks(cs_detect_x_idx * self.model_out_final_conv)[0]
        print("First detect CS, num: "+str(len(cs_detect_x_idx)))
        
        print("Re-align to local abs CS peak")
        # detect re-align to the local abs peak
        temp_idx = []
        for i_idx in range(len(cs_detect_x_idx)):
            if cs_detect_x_idx[i_idx]-int(self.fs_ms*5) < 0:
                continue
            if cs_detect_x_idx[i_idx]+int(self.fs_ms*5) >= self.L_data:
                continue
            temp = np.argmax(self.CS_peak_sign*(np.sum(self.ap[:,cs_detect_x_idx[i_idx]-int(self.fs_ms*5):cs_detect_x_idx[i_idx]+int(self.fs_ms*5)],axis=0)))
            # edge check
            if cs_detect_x_idx[i_idx] + (temp - int(self.fs_ms*self.CS_seg_ms[0])) - int(self.fs_ms*self.CS_seg_ms[0]) < 0:
                continue
            if cs_detect_x_idx[i_idx] + (temp - int(self.fs_ms*self.CS_seg_ms[0])) + int(self.fs_ms*self.CS_seg_ms[1]) >= self.L_data:
                continue
            temp_idx.append(cs_detect_x_idx[i_idx] + (temp - int(self.fs_ms*self.CS_seg_ms[0])))
        temp_idx = np.array(temp_idx)
        print("Num after re-align: "+str(len(temp_idx)))

        print("Remove double counting")
        # post detect remove double counting
        while np.any(np.diff(temp_idx) < self.fs_ms*1):
            temp = [np.inf]
            temp.extend(np.diff(temp_idx))
            temp = np.array(temp)
            temp_idx = temp_idx[temp>self.fs_ms*1]
        print("Num after remove double counting: "+str(len(temp_idx)))

        self.CS_detect_x_idx = np.array(temp_idx)

    def get_ACG_from_idx(self, x_idx):
        ap_acg = []
        lfp_acg = []
        new_x_idx = []
        for i_idx in range(len(x_idx)):
            
            if x_idx[i_idx]-int(self.fs_ms*10) < 0:
                continue
            if x_idx[i_idx]+int(self.fs_ms*10) >= self.L_data:
                continue
            
            new_x_idx.append(x_idx[i_idx])

            this_ap_acg = []
            this_lfp_acg = []
            for i_ch in range(self.channel_span.shape[0]):
                this_ap_acg.append(np.convolve(self.ap[i_ch,int(x_idx[i_idx]-self.fs_ms*5):int(x_idx[i_idx]+self.fs_ms*5)],self.ap[i_ch,int(x_idx[i_idx]-self.fs_ms*10):int(x_idx[i_idx]+self.fs_ms*10)],mode='valid'))
                this_lfp_acg.append(np.convolve(self.lfp[i_ch,int(x_idx[i_idx]-self.fs_ms*5):int(x_idx[i_idx]+self.fs_ms*5)],self.lfp[i_ch,int(x_idx[i_idx]-self.fs_ms*10):int(x_idx[i_idx]+self.fs_ms*10)],mode='valid'))
            ap_acg.append(torch.tensor(this_ap_acg).unsqueeze(0))
            lfp_acg.append(torch.tensor(this_lfp_acg).unsqueeze(0))
        ap_acg = torch.cat(ap_acg,dim=0)
        lfp_acg = torch.cat(lfp_acg,dim=0)

        return ap_acg,lfp_acg, np.array(new_x_idx)
    
    def get_waveform_from_idx(self, x_idx):
        ap_waveform = []
        lfp_waveform = []
        new_x_idx = []
        for i_idx in range(len(x_idx)):
            
            if x_idx[i_idx]-int(self.fs_ms*10) < 0:
                continue
            if x_idx[i_idx]+int(self.fs_ms*10) >= self.L_data:
                continue
            
            new_x_idx.append(x_idx[i_idx])

            ap_waveform.append(torch.tensor(self.ap[:,int(x_idx[i_idx]-self.fs_ms*5):int(x_idx[i_idx]+self.fs_ms*5)].copy()).unsqueeze(0))
            lfp_waveform.append(torch.tensor(self.lfp[:,int(x_idx[i_idx]-self.fs_ms*5):int(x_idx[i_idx]+self.fs_ms*5)].copy()).unsqueeze(0))
        ap_waveform = torch.cat(ap_waveform,dim=0)
        lfp_waveform = torch.cat(lfp_waveform,dim=0)

        return ap_waveform,lfp_waveform, np.array(new_x_idx)

    def clustering_from_acg(self):
        print("First clustering from ACG")
        # get ACG from CS_detect_x_idx
        ap_acg, lfp_acg, self.CS_detect_x_idx = self.get_ACG_from_idx(self.CS_detect_x_idx)

        acg_flatten_feature = torch.cat([(ap_acg[:,:,:]).unsqueeze(3),(lfp_acg[:,:,:]).unsqueeze(3)],axis=3)
        acg_flatten_feature = torch.flatten(acg_flatten_feature,start_dim=1,end_dim=-1)

        my_umap = umap.UMAP(n_neighbors=10,force_approximation_algorithm=True, n_jobs=-1)
        umap_acg_flatten_feature = my_umap.fit_transform(acg_flatten_feature)

        # DBSCAN clustering
        my_dbscan = DBSCAN(eps=self.DBSCAN_eps, min_samples=10, n_jobs=-1)
        my_dbscan.fit(umap_acg_flatten_feature)
        my_dbscan_clusters = np.array(my_dbscan.labels_)
        num_cluster = len(np.unique(my_dbscan_clusters))

        zero_embedding = np.zeros_like(acg_flatten_feature)+np.random.rand(*acg_flatten_feature.shape)
        zero_embedding = zero_embedding[:10,:]
        zero_embedding = my_umap.transform(zero_embedding)

        if self.enable_plot:
            # visualize clustering results
            scatter_color_list = ['y','b','g','c','m','orange','gold','y','b','g','c','m','orange','gold']

            plt.figure()
            plt.scatter(umap_acg_flatten_feature[:,0],umap_acg_flatten_feature[:,1],c='k',s=1)
            for i in range(num_cluster):
                if i < len(scatter_color_list):
                    plt.scatter(umap_acg_flatten_feature[my_dbscan_clusters==i,0],umap_acg_flatten_feature[my_dbscan_clusters==i,1],c=scatter_color_list[i],s=1)
            plt.title('UMAP')

            # plot 0 embedding with added random noise
            plt.scatter(zero_embedding[:,0],zero_embedding[:,1],c='k',s=100,marker='X')
            plt.show()

        self.CS_clustered_x_idx = []
        self.CS_peak_channel = []

        # stop here if no more than 1 cluster
        if num_cluster <= 1:
            print("Error: No complex spike cluster found! Check your signal!")
            return

        # return all clusters with the far distance from 0 embedding center
        # except the cluster that is too close to 0 embedding center
        zero_embedding_center = np.mean(zero_embedding,axis=0)
        cluster_distance = np.zeros((num_cluster,))
        for i in range(num_cluster):
            cluster_distance[i] = np.mean(np.sqrt((umap_acg_flatten_feature[my_dbscan_clusters==i,0]-zero_embedding_center[0])**2+(umap_acg_flatten_feature[my_dbscan_clusters==i,1]-zero_embedding_center[1])**2))
        furthest_cluster_list = np.argsort(cluster_distance)[::-1]

        for i_cluster in range(num_cluster-1):
            
            # if cluster number is too large and exceed 5Hz, it is clearly not a CS cluster
            if np.sum(my_dbscan_clusters==furthest_cluster_list[i_cluster]) > self.L_data/self.fs*5:
                continue 

            this_cluster_x_idx = self.CS_detect_x_idx[my_dbscan_clusters==furthest_cluster_list[i_cluster]]

            # determine which CS peak channel based on all square sum
            channel_square_sum = np.zeros((self.channel_span.shape[0],))
            for i_idx in range(len(this_cluster_x_idx)):
                for i_ch in range(self.channel_span.shape[0]):
                    channel_square_sum[i_ch] = channel_square_sum[i_ch] + np.sum(self.ap[i_ch,this_cluster_x_idx[i_idx]-int(self.fs_ms*5):this_cluster_x_idx[i_idx]+int(self.fs_ms*5)]**2)
            this_CS_peak_channel = np.argmax(channel_square_sum)

            

            # re-align among this cluster
            temp_idx = this_cluster_x_idx.copy()
            for i_idx in range(len(this_cluster_x_idx)):
                temp = np.argmax(np.convolve(np.abs(self.ap[this_CS_peak_channel,int(this_cluster_x_idx[i_idx]-self.fs_ms*5):int(this_cluster_x_idx[i_idx]+self.fs_ms*5)]),np.hamming(int(self.fs_ms*10)),mode='full'))
                temp_idx[i_idx] = this_cluster_x_idx[i_idx] + (temp - int(self.fs_ms*10))
            # all re-align to local peak
            temp2_idx = temp_idx.copy()
            for i_idx in range(len(temp2_idx)):
                temp = np.argmax(self.CS_peak_sign*self.ap[this_CS_peak_channel,int(temp_idx[i_idx]-self.fs_ms*10):int(temp_idx[i_idx]+self.fs_ms*10)])
                temp2_idx[i_idx] = temp_idx[i_idx] + (temp - int(self.fs_ms*10))
            temp_idx = temp2_idx.copy()

            # post detect remove double counting
            while np.any(np.diff(temp_idx) < self.fs_ms*1):
                temp = [np.inf]
                temp.extend(np.diff(temp_idx))
                temp = np.array(temp)
                temp_idx = temp_idx[temp>self.fs_ms*1]

                # all re-align to local peak
                temp2_idx = temp_idx.copy()
                for i_idx in range(len(temp2_idx)):
                    temp = np.argmax(self.CS_peak_sign*self.ap[this_CS_peak_channel,int(temp_idx[i_idx]-self.fs_ms*10):int(temp_idx[i_idx]+self.fs_ms*10)])
                    temp2_idx[i_idx] = temp_idx[i_idx] + (temp - int(self.fs_ms*10))
                temp_idx = temp2_idx.copy()

            self.CS_peak_channel.append(this_CS_peak_channel+self.channel_span[0])
            self.CS_clustered_x_idx.append(np.array(temp_idx))
            print("Detect CS cluster on channel: "+str(this_CS_peak_channel+self.channel_span[0])+", num: "+str(len(temp_idx)))

    def clustering_from_waveform(self):
        print("First clustering from waveform")
        # get waveform from CS_detect_x_idx
        ap_waveform, lfp_waveform, self.CS_detect_x_idx = self.get_waveform_from_idx(self.CS_detect_x_idx)

        acg_flatten_feature = torch.cat([(ap_waveform[:,:,:]).unsqueeze(3),(lfp_waveform[:,:,:]).unsqueeze(3)],axis=3)
        acg_flatten_feature = torch.flatten(acg_flatten_feature,start_dim=1,end_dim=-1)

        print("Fit UMAP")
        my_umap = umap.UMAP(n_neighbors=10,force_approximation_algorithm=True, n_jobs=-1)
        umap_acg_flatten_feature = my_umap.fit_transform(acg_flatten_feature)

        # DBSCAN clustering
        print("DBSCAN clustering")
        my_dbscan = DBSCAN(eps=self.DBSCAN_eps, min_samples=10, n_jobs=-1)
        my_dbscan.fit(umap_acg_flatten_feature)
        my_dbscan_clusters = np.array(my_dbscan.labels_)
        num_cluster = len(np.unique(my_dbscan_clusters))

        print("DBSCAN zero embedding")
        zero_embedding = np.zeros_like(acg_flatten_feature)+np.random.rand(*acg_flatten_feature.shape)
        zero_embedding = zero_embedding[:10,:]
        zero_embedding = my_umap.transform(zero_embedding)

        if self.enable_plot:
            # visualize clustering results
            scatter_color_list = ['y','b','g','c','m','orange','gold','y','b','g','c','m','orange','gold']

            plt.figure()
            plt.scatter(umap_acg_flatten_feature[:,0],umap_acg_flatten_feature[:,1],c='k',s=1)
            for i in range(num_cluster):
                if i < len(scatter_color_list):
                    plt.scatter(umap_acg_flatten_feature[my_dbscan_clusters==i,0],umap_acg_flatten_feature[my_dbscan_clusters==i,1],c=scatter_color_list[i],s=1)
            plt.title('UMAP')

            # plot 0 embedding with added random noise
            plt.scatter(zero_embedding[:,0],zero_embedding[:,1],c='k',s=100,marker='X')
            plt.show()

        self.CS_clustered_x_idx = []
        self.CS_peak_channel = []

        # stop here if no more than 1 cluster
        if num_cluster <= 1:
            print("Error: No complex spike cluster found! Check your signal!")
            return

        # return all clusters with the far distance from 0 embedding center
        # except the cluster that is too close to 0 embedding center
        zero_embedding_center = np.mean(zero_embedding,axis=0)
        cluster_distance = np.zeros((num_cluster,))
        for i in range(num_cluster):
            cluster_distance[i] = np.mean(np.sqrt((umap_acg_flatten_feature[my_dbscan_clusters==i,0]-zero_embedding_center[0])**2+(umap_acg_flatten_feature[my_dbscan_clusters==i,1]-zero_embedding_center[1])**2))
        furthest_cluster_list = np.argsort(cluster_distance)[::-1]

        for i_cluster in range(num_cluster-1):
            
            # if cluster number is too large and exceed 5Hz, it is clearly not a CS cluster
            if np.sum(my_dbscan_clusters==furthest_cluster_list[i_cluster]) > self.L_data/self.fs*5:
                continue 

            this_cluster_x_idx = self.CS_detect_x_idx[my_dbscan_clusters==furthest_cluster_list[i_cluster]]

            # determine which CS peak channel based on all square sum
            channel_square_sum = np.zeros((self.channel_span.shape[0],))
            for i_idx in range(len(this_cluster_x_idx)):
                for i_ch in range(self.channel_span.shape[0]):
                    channel_square_sum[i_ch] = channel_square_sum[i_ch] + np.sum(self.ap[i_ch,this_cluster_x_idx[i_idx]-int(self.fs_ms*5):this_cluster_x_idx[i_idx]+int(self.fs_ms*5)]**2)
            this_CS_peak_channel = np.argmax(channel_square_sum)

            

            # re-align among this cluster
            temp_idx = this_cluster_x_idx.copy()
            for i_idx in range(len(this_cluster_x_idx)):
                temp = np.argmax(np.convolve(np.abs(self.ap[this_CS_peak_channel,int(this_cluster_x_idx[i_idx]-self.fs_ms*5):int(this_cluster_x_idx[i_idx]+self.fs_ms*5)]),np.hamming(int(self.fs_ms*10)),mode='full'))
                temp_idx[i_idx] = this_cluster_x_idx[i_idx] + (temp - int(self.fs_ms*10))
            # all re-align to local peak
            temp2_idx = temp_idx.copy()
            for i_idx in range(len(temp2_idx)):
                temp = np.argmax(self.CS_peak_sign*self.ap[this_CS_peak_channel,int(temp_idx[i_idx]-self.fs_ms*10):int(temp_idx[i_idx]+self.fs_ms*10)])
                temp2_idx[i_idx] = temp_idx[i_idx] + (temp - int(self.fs_ms*10))
            temp_idx = temp2_idx.copy()

            # post detect remove double counting
            while np.any(np.diff(temp_idx) < self.fs_ms*1):
                temp = [np.inf]
                temp.extend(np.diff(temp_idx))
                temp = np.array(temp)
                temp_idx = temp_idx[temp>self.fs_ms*1]

                # all re-align to local peak
                temp2_idx = temp_idx.copy()
                for i_idx in range(len(temp2_idx)):
                    temp = np.argmax(self.CS_peak_sign*self.ap[this_CS_peak_channel,int(temp_idx[i_idx]-self.fs_ms*10):int(temp_idx[i_idx]+self.fs_ms*10)])
                    temp2_idx[i_idx] = temp_idx[i_idx] + (temp - int(self.fs_ms*10))
                temp_idx = temp2_idx.copy()

            self.CS_peak_channel.append(this_CS_peak_channel+self.channel_span[0])
            self.CS_clustered_x_idx.append(np.array(temp_idx))
            print("Detect CS cluster on channel: "+str(this_CS_peak_channel+self.channel_span[0])+", num: "+str(len(temp_idx)))

    def forward(self):
        self.load_model()
        self.load_data()

        self.preprocess()
        self.model_predict()
        self.model_out_conv()
        self.detect_CS_idx()
        if self.if_cluster_waveform:
            self.clustering_from_waveform()
        else:
            self.clustering_from_acg()
        print("Done forward!")

    def forward_CS_anchor_list(self, CS_anchor_list):

        self.CS_anchor_list = CS_anchor_list
        self.load_model()
        self.load_data()

        # save CS_peak_channel and CS_clustered_x_idx
        self.CS_peak_channel_list = []
        self.CS_clustered_x_idx_list = []
        for i_anchor in range(len(CS_anchor_list)):
            self.ch_anchor = CS_anchor_list[i_anchor]
            print("CS anchor: "+str(self.ch_anchor))
            self.preprocess()
            self.model_predict()
            self.model_out_conv()
            self.detect_CS_idx()
            if self.if_cluster_waveform:
                self.clustering_from_waveform()
            else:
                self.clustering_from_acg()
            self.CS_peak_channel_list.append(self.CS_peak_channel)
            self.CS_clustered_x_idx_list.append(self.CS_clustered_x_idx)
        print("Done forward_CS_anchor_list!")



