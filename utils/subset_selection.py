"""
Code for subset selection referred from:
https://github.com/hbdat/eccv20_Multi_Task_Procedure_Learning/blob/main/core/
self_supervision_summarization.py
We thank the authors for releasing the code!
"""

import torch
import scipy
import numpy as np
from torch import nn
from sklearn.cluster import MiniBatchKMeans


class SelfSupervisionSummarization(nn.Module):
    def __init__(self, M, repNum, dim=128):
        super(SelfSupervisionSummarization, self).__init__()
        self.M = M
        self.dim = dim
        self.flush()
        self.kmeans = MiniBatchKMeans(
            n_clusters=M,
            init='k-means++',
            max_iter=1000,
            n_init=50,
            random_state=0,
            batch_size=256*20
        )
        self.repNum = repNum

    def flush(self):
        # data buffer that hold all the aggregate attention feature of each
        # video segment
        self.list_fbar_seg = []
        self.list_labels = []
        self.list_video_lens = []
        self.dict_video_name = {}
        self.v_counter = 0

    def add_video(self, fbar_seg, video_name):
        if fbar_seg.shape[-1] != self.dim:
            fbar_seg = torch.transpose(fbar_seg,1,0)
        
        assert fbar_seg.shape[-1] == self.dim
        
        self.list_fbar_seg.append(fbar_seg)
        self.list_video_lens.append(fbar_seg.shape[0])
        if video_name in self.dict_video_name:
            raise Exception('Reload a video twice {}'.format(video_name))
        self.dict_video_name[video_name] = self.v_counter
        self.v_counter += 1

    def get_key_step_label(self, video_name):
        idx = self.dict_video_name[video_name]
        return self.list_labels[idx]

    def forward(self):
        # All the features from the videos
        all_fbar = np.concatenate(self.list_fbar_seg,axis = 0)
        Y = all_fbar
        # Apply kmeans to data to get centroids
        self.kmeans.fit(Y)
        # X is the M x d array of M centers in d-dimension
        X = self.kmeans.cluster_centers_
        # Compute similarity between X and Y
        S = -scipy.spatial.distance.cdist(X, Y, metric='euclidean')

        # Run subset selection
        # repNum: number of representative centers
        # reps: representative centers
        # assignments: assignments of segments to representative centers
        self.reps, self.assignments = self.run_ss(S,self.repNum)        
        all_keystep_labels = self.reps[self.assignments]

        assert len(self.list_labels) == 0
        accum_len = 0
        for l in self.list_video_lens:
            step_key_label = all_keystep_labels[accum_len:accum_len+l]
            assert step_key_label.shape[0] == l
            ## format back to torch
            step_key_label = torch.from_numpy(step_key_label[np.newaxis])
            ## format back to torch
            self.list_labels.append(step_key_label)
            accum_len += l
        return (
            torch.from_numpy(self.reps),
            torch.from_numpy(all_keystep_labels)
        )

    def predict(self,fbar,cat=-1):
        if len(fbar.size()) == 3:       #[1,n,d]
            fbar = fbar[0]
        if fbar.size(-1) != self.dim:
            fbar = torch.transpose(fbar,1,0)        
        assert fbar.size(-1) == self.dim        
        fbar= fbar.cpu().numpy()
        Y = fbar
        # Get centroid without applying kmeans
        # X is the M x d array of M centers in d-dimension
        X = self.kmeans.cluster_centers_

        # Compute similarity between X and Y
        S = -scipy.spatial.distance.cdist(X, Y, metric='euclidean')

        # Run subset selection
        # repNum: number of representative centers
        # reps: representative centers
        # assignments: assignments of segments to representative centers
        cost, assgn = self.ss_cost(S, self.reps)
        keystep_labels = self.reps[assgn]
        return (
            torch.from_numpy(self.reps),
            torch.from_numpy(keystep_labels[np.newaxis])
        )

    ################# Ehsan code #################
    # Funtion that takes the similarity matrix between Kmeans centers and
    # segment features and returns the set of representative centers and
    # assignments to representatives
    # S: similarity matrix between X and Y
    # repNum: number of representatives from X
    def run_ss(self,S,repNum):
        N = S.shape[0]
        active_set = np.empty(0)
        remaining_set = np.array(list(set(range(N)) - set(active_set)))
        cost1 = -float('inf')
        best_cost = -float('inf')
        assignment = np.array([0, N])
        for iter in range(repNum):
            for i in range(len(remaining_set)):
                element = remaining_set[i]
                [cost2, assignment2] = self.ss_cost(
                    S,
                    np.append(active_set,element).astype(int)
                )
                if (cost2 > best_cost):
                    best_cost = cost2
                    best_index = element
                    best_assignment = assignment2
            if (best_cost > cost1):
                active_set = np.append(active_set, best_index)
                remaining_set = np.array(list(set(range(N)) - set(active_set)))
                cost1 = best_cost
                assignment = best_assignment
            else:
                break
        return active_set.astype(int), assignment.astype(int)

    # Function to compute the best assignment for a given active set
    # S: similarity matrix between X and Y
    # aset: subset of indices from X
    def ss_cost(self,S, aset):
        N = S.shape[0]
        #[v, assgn] = torch.max(S[aset,:],0)
        v = np.ndarray.max(S[aset,:], 0)
        assgn = np.ndarray.argmax(S[aset,:], 0)
        #cost = sum(v).detach().numpy()
        cost = sum(v)
        return cost, assgn
    ################# Ehsan code #################
