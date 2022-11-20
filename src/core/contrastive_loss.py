import numpy as np
import torch
from torch import nn


# The Contrastive Loss implementation is taken from https://github.com/LishinC/VCN
class ContrastiveLoss(nn.Module):
    """
    TODO update this text
    Phase contrastive loss / Volume contrastive loss if custom_margin assigned
    accepts model 2 outputs (ED & ES), both of shape [B, embed_dim=128, 2].
    The 2nd-dimension contains the N-dimensional vector embeddings for frames,
    namely A4C ED, A2C ED in the first array; A4C ES, A2C ES in the second array.
    The loss encourage similar embeddings for the same phase, regardless of view.
    The positive pairs are the  ES pair and the ED pair.
    The negative pairs are the A4C pair and the A2C pair.
    """
    def __init__(self, default_margin=0., ed_adj_count=2, es_adj_count=3):
        super().__init__()
        self.default_margin = default_margin
        self.ed_adj_count = ed_adj_count
        self.es_adj_count = es_adj_count

    def forward(self, data, embeddings, batch_size=3, num_clips_per_video=3, custom_margin=None):
        d_positive = torch.tensor(0., device=data.ed_frame.device)
        d_negative = torch.tensor(0., device=data.ed_frame.device)
        for i in range(0, batch_size):
            ed_frame = data[i].ed_frame.item()
            es_frame = data[i].es_frame.item()
            for j in range(num_clips_per_video):
                if ed_frame not in data[i].frame_idx[j] or es_frame not in data[i].frame_idx[j]:
                    continue

                # indices of the adjacent frames to ED
                adj_ed_emb_idx = self.get_adj_embedding_indices(frame_idx=data[i].frame_idx[j],
                                                                anchor_idx=ed_frame,
                                                                count=self.ed_adj_count)
                # extract the embeddings into a new tensor
                adj_ed_embeddings = torch.stack([embeddings[i*num_clips_per_video + j, col, :]
                                                  for col in adj_ed_emb_idx]).squeeze()
                ed_embedding_col = self.get_anchor_idx_from_clip(frame_idx=data[i].frame_idx[j],
                                                                 anchor_idx=ed_frame)

                ed_embedding = embeddings[i*num_clips_per_video + j, ed_embedding_col]
                d_positive += self.calculate_distance(anchor=ed_embedding,
                                                      target=adj_ed_embeddings)
                adj_es_embedd_idx = self.get_adj_embedding_indices(frame_idx=data[i].frame_idx[j],
                                                                   anchor_idx=es_frame,
                                                                   count=self.es_adj_count)
                es_embedding_col = self.get_anchor_idx_from_clip(frame_idx=data[i].frame_idx[j],
                                                                 anchor_idx=es_frame)
                es_embedding = embeddings[i*num_clips_per_video + j, es_embedding_col]
                adj_es_embeddings = torch.stack([embeddings[i * num_clips_per_video + j, col, :]
                                                 for col in adj_es_embedd_idx]).squeeze()
                # calculate the distance of anchors and adjacent embeddings
                d_positive += self.calculate_distance(anchor=es_embedding,
                                                      target=adj_es_embeddings)

                d_negative = self.calculate_distance(anchor=ed_embedding, target=es_embedding)

        # assert  outputED.shape[1:] == (128,2)
        if custom_margin is None:
            custom_margin = self.default_margin
        else:
            custom_margin[custom_margin>1] = 1      #For volume contrastive loss, constrain margin to less than 1
        # d_positive, d_negative = 0, 0

        loss = torch.clamp(custom_margin + d_positive - d_negative, min=0.0).mean()
        # print('d_positive d_negative', d_positive, d_negative)
        return loss

    def distance(self, a, b, channel_dim=1):
        diff = torch.abs(a - b)
        return torch.pow(diff, 2).sum(dim=channel_dim) #[B, num_pairs=1]

    def get_adj_embedding_indices(self, frame_idx: list, anchor_idx: int, count: int):
        """
        Returns the indices of the adj frames of an anchor frame in the frame_idx

        :param frame_idx: list, list of frame indices for each clip
        :param anchor_idx: int, index of the anchor frame whose neighbors are gathered
        :param count: int, the number of frames on each side of the anchor frame to gather

        """

        # check whether if labeled ED or ES frame exist in the miniclip
        if not anchor_idx in frame_idx:
            return []
        # Need to ensure that the range here is within the indices of the video
        start_frame, end_frame = np.min(frame_idx), np.max(frame_idx)
        # if anchor_idx + count > end_frame:
        #     print(f"Requested count: {count} with anchor index: {anchor_idx} exceeds"
        #           f" the length of the video {end_frame}.")
        # if anchor_idx - count < start_frame:
        #     print(f"Requested count: {count} with anchor index: {anchor_idx} exceeds"
        #           f" the beginning of the video {start_frame}.")
        min_idx = max(start_frame, anchor_idx - count)
        max_idx = min(anchor_idx + count, end_frame)
        col_idx = np.where((frame_idx >= min_idx) & (frame_idx <= max_idx) & (frame_idx != anchor_idx))
        return col_idx

    def get_anchor_idx_from_clip(self, frame_idx: list, anchor_idx: int) -> np.array:
        return np.where(anchor_idx == frame_idx)

    def calculate_distance(self, anchor: torch.tensor, target: torch.tensor):
        distance = 0
        for embedding in target:
            distance += self.distance(a=anchor, b=embedding)
        return distance.mean()
