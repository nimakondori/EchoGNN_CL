import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# The Contrastive Loss implementation based on https://github.com/LishinC/VCN
class ContrastiveLoss(nn.Module):
    """
    ContrastiveLoss class implements the loss on the given anchors (ED and ES) frames
    and their neighboring frames, caculates the loss and returns the result

    Attributes
    ----------
    ed_adj_count: int, the number of ed neighboring frames to cluster
    es_adj_count: int, the number of es neighboring frames to cluster
    volume_distance : bool, if we should apply volumetric margin or not
    volume_scaling: int, the scaling factor to center the data around a target

    Methods
    -------
    forward(self, data, embeddings, batch_size, num_clips_per_video): calculates the loss per batch
    distance(self, a: Tensor, b: Tensor): calculates the norm squared distance between 2 vectors
    get_adj_embedding_indices(self, frame_idx: list, anchor_idx: int, count: int): finds the indices around the
    anchor frame based on indices in frame indices in each clip
    get_anchor_idx_from_clip(frame_idx: list, anchor_idx: int): returns the index of the anchor within a video clip
    calculate_distance(self, anchor: Tensor, target: Tensor, custom_margin: float): calculate the distance between
    clusters (d_neg)
    calculate_distance2(self, embeddings: Tensor, anchor_idx: tuple, target_idx: list): calculate the distance between
    the anchor and the neighboring frames (d_pos)

    """

    def __init__(self,
                 ed_adj_count: int = 2,
                 es_adj_count: int = 3,
                 volume_distance: bool = False, volume_scaling: int = 200):
        super().__init__()
        self.ed_adj_count = ed_adj_count
        self.es_adj_count = es_adj_count
        self.volume_scaling = volume_scaling
        self.volume_distance = volume_distance

    def forward(self,
                data,
                embeddings,
                batch_size=3,
                num_clips_per_video=3):

        loss = torch.tensor(0., device=data.ed_frame.device)

        for i in range(0, batch_size):
            ed_frame = data[i].ed_frame.item()
            es_frame = data[i].es_frame.item()
            edv = data[i].edv.item()
            esv = data[i].esv.item()

            # custom margin is the volume based distance factor of ED and ES
            custom_margin = (edv - esv) / self.volume_scaling if self.volume_distance else 0

            for j in range(num_clips_per_video):
                d_positive = torch.tensor(0., device=data.ed_frame.device)
                d_negative = torch.tensor(0., device=data.ed_frame.device)
                # This differentiates between frame_idx shape of (64, ) and other shapes
                frame_idx = data[i].frame_idx[j] if len(data[i].frame_idx.shape) > 1 \
                    else data[i].frame_idx

                # We skip the clips that don't include both ED and ES frames
                if ed_frame not in frame_idx or es_frame not in frame_idx:
                    continue

                # indices of the adjacent frames to ED
                adj_ed_emb_idx = self.get_adj_embedding_indices(frame_idx=frame_idx,
                                                                anchor_idx=ed_frame,
                                                                count=self.ed_adj_count)
                ed_embedding_col = self.get_anchor_idx_from_clip(frame_idx=frame_idx,
                                                                 anchor_idx=ed_frame)

                d_positive += self.calculate_distance2(
                    embeddings=embeddings,
                    anchor_idx=(j, ed_embedding_col),
                    target_idx=adj_ed_emb_idx)

                # indices of the adjacent frames to ES
                adj_es_emb_idx = self.get_adj_embedding_indices(frame_idx=frame_idx,
                                                                anchor_idx=es_frame,
                                                                count=self.es_adj_count)
                es_embedding_col = self.get_anchor_idx_from_clip(frame_idx=frame_idx,
                                                                 anchor_idx=es_frame)
                # calculate the distance of anchors and adjacent embeddings
                d_positive += self.calculate_distance2(
                    embeddings=embeddings,
                    anchor_idx=(j, es_embedding_col),
                    target_idx=adj_es_emb_idx)

                d_negative = self.calculate_distance(anchor=embeddings[i * num_clips_per_video + j, ed_embedding_col],
                                                     target=embeddings[i * num_clips_per_video + j, es_embedding_col],
                                                     custom_margin=custom_margin)

                # d_positive / 2 since it is being added from 2 sources as opposed to d_negative
                loss += torch.clamp(custom_margin + d_positive - d_negative, min=0.0).mean()

        return loss / batch_size

    def distance(self, a: Tensor, b: Tensor):
        """
        Calculates the distance between the input vectors

        :param a: torch.Tensor, input vector 1
        :param b: torch.Tensor, input vector 2

        :returns: norm squared between the input vectors
        """
        diff = torch.abs(a - b)
        return (torch.pow(diff, 2)).sum()

    def get_adj_embedding_indices(self, frame_idx: list, anchor_idx: int, count: int):
        """
        Finds the location of the neighboring indices of the anchor within video clip

        :param frame_idx: list, list of frame indices for each clip
        :param anchor_idx: int, index of the anchor frame whose neighbors are gathered
        :param count: int, the number of frames on each side of the anchor frame to gather

        :return: the indices of the adj frames of an anchor frame in the frame_idx
        """

        # check whether if labeled ED or ES frame exist in the miniclip
        if not anchor_idx in frame_idx:
            return []
        # Need to ensure that the range here is within the indices of the video
        start_frame, end_frame = np.min(frame_idx), np.max(frame_idx)
        min_idx = max(start_frame, anchor_idx - count)
        max_idx = min(anchor_idx + count, end_frame)
        col_idx = np.where((frame_idx >= min_idx) & (frame_idx <= max_idx) & (frame_idx != anchor_idx))
        # remove the data information and return
        return col_idx[0]

    @staticmethod
    def get_anchor_idx_from_clip(frame_idx: list, anchor_idx: int) -> list:
        """
        Finds the location of the anchor withing a video clip

        :param frame_idx: list, list of frame indices for each clip
        :param anchor_idx: int, index of the anchor frame whose neighbors are gathered

        :return: the index of the anchor frame withing each video clip
        """

        # only return the idx rather than the array
        return np.where(anchor_idx == frame_idx)[0][0]

    def calculate_distance(self, anchor: Tensor, target: Tensor):
        """
        Calculate the distance between 2  clusters by finding the distance between the 2 anchors

         :param anchor: torch.Tensor, anchor frame embedding which is the ED frame
         :param target: torch.Tensor, target frame embedding which is the ES frame

         :returns: the distance between 2 cluster
         """
        distance = 0

        distance += self.distance(a=F.normalize(anchor, p=2, dim=0),
                                  b=F.normalize(target, p=2, dim=0))
        return distance

    def calculate_distance2(self, embeddings: Tensor, anchor_idx: tuple, target_idx: list):
        """
        Finds the average distance of positive samples and the anchor

         :param embeddings: torch.Tensor, a tensor including the embeddings of a video clip
         :param anchor_idx: tuple, index of the anchor withing the video clip
         :param target_idx: list, indices of the target frames which are the neighboring frames

         :returns: the average distance between the neighboring frames and the anchor frame
         """
        distance = 0
        # This function is called on each j so the row of the embeddings and the anchor is the same
        target_row = anchor_idx[0]
        for col in target_idx:
            distance += self.distance(a=F.normalize(embeddings[anchor_idx], p=2, dim=0),
                                      b=F.normalize(embeddings[target_row, col], p=2, dim=0))
        # return the average to keep the scale of negative and positive distances the same
        return distance / len(target_idx)
