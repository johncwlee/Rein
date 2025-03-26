from mmseg.models.decode_heads.mask2former_head import Mask2FormerHead
from mmseg.registry import MODELS
from mmseg.utils import SampleList
from torch import Tensor
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from mmseg.models.builder import MODELS
from mmseg.utils import ConfigType, SampleList
import torch.nn.functional as F


@MODELS.register_module()
class ReinMask2FormerHead(Mask2FormerHead):
    def __init__(self, replace_query_feat=False, **kwargs):
        super().__init__(**kwargs)
        feat_channels = kwargs["feat_channels"]
        del self.query_embed
        self.vpt_transforms = nn.ModuleList()
        self.replace_query_feat = replace_query_feat
        if replace_query_feat:
            del self.query_feat
            self.querys2feat = nn.Linear(feat_channels, feat_channels)

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, query_embed = x
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)
        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

@MODELS.register_module()
class ReinMask2FormerHead2(ReinMask2FormerHead):
    def __init__(self, loss_con: ConfigType, **kwargs):
        super().__init__(**kwargs)
        self.ood_index = loss_con.ood_idx
        self.temperature = loss_con.temperature
        self.con_loss_weight = loss_con.loss_weight
        self.max_views = 512

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, query_embed = x
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)
        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        if self.training:
            return x, cls_pred_list, mask_pred_list
        else:
            return cls_pred_list, mask_pred_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        gt_mask_list = []
        # batch SegDataSample to InstanceDataSample
        for data_sample in batch_data_samples:
            gt_mask = (data_sample.gt_sem_seg.data == self.ood_index).long()
            gt_mask_list.append(gt_mask)
            data_sample.gt_sem_seg.data[data_sample.gt_sem_seg.data == self.ood_index] = 255

        gt_masks = torch.cat(gt_mask_list)
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        feats, all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                batch_gt_instances, batch_img_metas)
        
        #* Contrastive loss
        feats = feats[-1]   #* (B, C, H, W)
        gt_masks = F.interpolate(gt_masks.unsqueeze(1).float(), 
                    size=feats.shape[2:],
                    mode='nearest').squeeze(1).long()
        embeds = F.normalize(feats, p=2, dim=1)
        
        #? Randomly sample the anchor and contrastive samples
        anchor_embeds, anchor_labels = self.get_anchor_con_samples(embeds, gt_masks)

        #? calculate the InfoNCE
        if anchor_labels.numel() > 2:
            con_loss = self.info_nce(a_samples=anchor_embeds, 
                                    a_labels=anchor_labels.unsqueeze(1), 
                                    c_samples=anchor_embeds,
                                    c_labels=anchor_labels.unsqueeze(1))
        else:
            con_loss = torch.tensor(0., device=embeds.device)

        losses["loss_con"] = self.con_loss_weight * con_loss
        return losses

    def info_nce(self, a_samples, a_labels, c_samples, c_labels) -> torch.Tensor:
        """Calculates the Noise Contrastive Estimation loss.

        Args:
            a_samples (_type_): Anchor set with shape (N, D)
            a_labels (_type_): Anchor set labels with shape (N, 1)
            c_samples (_type_): Contrastive set with shape (M, D)
            c_labels (_type_): Contrastive set labels with shape (M, 1)

        Returns:
            torch.Tensor: Contrastive loss.
        """
        #? calculates the binary mask: same category => 1, different categories => 0
        mask = torch.eq(a_labels, c_labels.transpose(0, 1)).float()

        #? compute temperature-scaled dot product between anchor and contrastive samples
        anchor_dot_contrast = (a_samples @ c_samples.transpose(0,1)) / self.temperature

        #* for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        #? calculate the negative mask
        neg_mask = 1 - mask
        
        #? avoid the self duplicate issue
        mask = mask.fill_diagonal_(0.)

        #? sum the negative odot results
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        #* log_prob -> log(exp(x))-log(exp(x) + exp(y))
        #* log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        log_prob = logits - torch.log(exp_logits + neg_logits)

        #? calculate the info-nce based on the positive samples (under same categories)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()

    def get_anchor_con_samples(self, embeds, labels):
        #? Reformat for ease of sampling
        embeds = embeds.flatten(start_dim=2).permute(0, 2, 1)
        labels = labels.flatten(start_dim=1)

        #? define different types of embeds
        pos = embeds[labels == 1]
        neg = embeds[labels == 0]

        num_sample_list = [self.max_views, pos.shape[0], neg.shape[0]]

        #? define the number of samples
        sample_num = int(min(*num_sample_list))

        #? randomly extract the anchor set
        anchor_pos = pos[torch.randperm(pos.shape[0])][:sample_num]
        anchor_neg = neg[torch.randperm(neg.shape[0])][:sample_num]
        anchor_set = torch.cat([anchor_pos, anchor_neg], dim=0)
        anchor_label = torch.cat([torch.ones(anchor_pos.shape[0], device=anchor_pos.device),
                                  torch.zeros(anchor_neg.shape[0], device=anchor_neg.device)])

        return anchor_set, anchor_label


@MODELS.register_module()
class ReinMask2FormerHead3(ReinMask2FormerHead):
    def __init__(self, loss_con: ConfigType, **kwargs):
        super().__init__(**kwargs)
        self.ood_index = loss_con.ood_idx
        self.temperature = loss_con.temperature
        self.con_loss_weight = loss_con.loss_weight
        self.max_views = 512

    def forward(
        self, x: Tuple[List[Tensor], List[Tensor]], batch_data_samples: SampleList
    ) -> Tuple[List[Tensor]]:
        x, query_embed = x
        batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        batch_size = len(batch_img_metas)
        if query_embed.ndim == 2:
            query_embed = query_embed.expand(batch_size, -1, -1)
        # use vpt_querys to replace query_embed
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size,) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool
            )
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2
            ).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        if self.replace_query_feat:
            query_feat = self.querys2feat(query_embed)
        else:
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        # query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:]
        )
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat,
                mask_features,
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[
                    -2:
                ],
            )

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        if self.training:
            return x, cls_pred_list, mask_pred_list
        else:
            return cls_pred_list, mask_pred_list

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        gt_mix_mask_list = []
        gt_ood_mask_list = []
        # batch SegDataSample to InstanceDataSample
        for data_sample in batch_data_samples:
            gt_mix_mask = (data_sample.gt_sem_seg.data == self.ood_index).long()
            gt_mix_mask_list.append(gt_mix_mask)
            data_sample.gt_sem_seg.data[data_sample.gt_sem_seg.data == self.ood_index] = 255
            
            gt_ood_mask = (data_sample.coco_gt_sem_seg.data == 1).long()
            gt_ood_mask_list.append(gt_ood_mask)

        gt_mix_masks = torch.cat(gt_mix_mask_list)
        gt_ood_masks = torch.cat(gt_ood_mask_list)
        batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
            batch_data_samples)

        # forward
        feats, query_embed = x
        B = len(batch_data_samples)
        feats_in = [f[:B] for f in feats]
        _, all_cls_scores, all_mask_preds = self((feats_in, query_embed), batch_data_samples)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
                                batch_gt_instances, batch_img_metas)
        
        #* Contrastive loss
        feats = feats[-1]   #* (B, C, H, W)
        gt_mix_masks = F.interpolate(gt_mix_masks.unsqueeze(1).float(), 
                        size=feats.shape[2:],
                        mode='nearest').squeeze(1).long()
        gt_ood_masks = F.interpolate(gt_ood_masks.unsqueeze(1).float(), 
                        size=feats.shape[2:],
                        mode='nearest').squeeze(1).long()
        embeds = F.normalize(feats, p=2, dim=1)
        
        #? Randomly sample the anchor and contrastive samples
        embeds_mix, embeds_ood = embeds[:B], embeds[B:]
        (anchor_embeds, anchor_labels,
         contrast_embeds, contrast_labels) = self.get_anchor_con_samples(embeds_mix, gt_mix_masks,
                                                                        embeds_ood, gt_ood_masks)

        #? calculate the InfoNCE
        if anchor_labels.numel() > 2:
            con_loss = self.info_nce(a_samples=anchor_embeds, 
                                    a_labels=anchor_labels.unsqueeze(1), 
                                    c_samples=contrast_embeds,
                                    c_labels=contrast_labels.unsqueeze(1))
        else:
            con_loss = torch.tensor(0., device=embeds.device)

        losses["loss_con"] = self.con_loss_weight * con_loss
        return losses

    def info_nce(self, a_samples, a_labels, c_samples, c_labels) -> torch.Tensor:
        """Calculates the Noise Contrastive Estimation loss.

        Args:
            a_samples (_type_): Anchor set with shape (N, D)
            a_labels (_type_): Anchor set labels with shape (N, 1)
            c_samples (_type_): Contrastive set with shape (M, D)
            c_labels (_type_): Contrastive set labels with shape (M, 1)

        Returns:
            torch.Tensor: Contrastive loss.
        """
        #? calculates the binary mask: same category => 1, different categories => 0
        mask = torch.eq(a_labels, c_labels.transpose(0, 1)).float()

        #? compute temperature-scaled dot product between anchor and contrastive samples
        anchor_dot_contrast = (a_samples @ c_samples.transpose(0,1)) / self.temperature

        #* for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        #? calculate the negative mask
        neg_mask = 1 - mask
        
        #? avoid the self duplicate issue
        mask = mask.fill_diagonal_(0.)

        #? sum the negative odot results
        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        #* log_prob -> log(exp(x))-log(exp(x) + exp(y))
        #* log_prob -> log{exp(x)/[exp(x)+exp(y)]}
        log_prob = logits - torch.log(exp_logits + neg_logits)

        #? calculate the info-nce based on the positive samples (under same categories)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        return -mean_log_prob_pos.mean()

    def get_anchor_con_samples(self, oe_embeds, oe_labels, ood_embeds, ood_labels):
        #? Reformat for ease of sampling
        oe_embeds = oe_embeds.flatten(start_dim=2).permute(0, 2, 1)
        oe_labels = oe_labels.flatten(start_dim=1)
        ood_embeds = ood_embeds.flatten(start_dim=2).permute(0, 2, 1)
        ood_labels = ood_labels.flatten(start_dim=1)

        #? define different types of embeds
        oe_pos = oe_embeds[oe_labels == 1]
        oe_neg = oe_embeds[oe_labels == 0]
        ood_pos = ood_embeds[ood_labels == 1]
        ood_neg = ood_embeds[ood_labels == 0]

        #? check if ood_neg is empty (probably if background of ood image is ignored explicitly)
        has_ood_neg = ood_neg.numel() > 0
        num_sample_list = [self.max_views, oe_pos.shape[0], oe_neg.shape[0], ood_pos.shape[0]]

        if has_ood_neg:
            num_sample_list.append(ood_neg.shape[0])

        #? define the number of samples
        sample_num = int(min(*num_sample_list))

        #? randomly extract the anchor set with {oe_pos, oe_neg}
        anchor_pos = oe_pos[torch.randperm(oe_pos.shape[0])][:sample_num]
        anchor_neg = oe_neg[torch.randperm(oe_neg.shape[0])][:sample_num]
        anchor_set = torch.cat([anchor_pos, anchor_neg], dim=0)
        anchor_label = torch.cat([torch.ones(anchor_pos.shape[0], device=anchor_pos.device),
                                  torch.zeros(anchor_neg.shape[0], device=anchor_neg.device)])

        #? randomly extract the contrastive set with {oe_pos, oe_neg, ood_pos, ood_neg}
        ood_pos_con = ood_pos[torch.randperm(ood_pos.shape[0])][:sample_num]
        ood_neg_con = ood_neg[torch.randperm(ood_neg.shape[0])][:sample_num]
        contrast_set = torch.cat([anchor_set.clone(), ood_pos_con, ood_neg_con], dim=0)
        contrast_label = torch.cat([anchor_label.clone(),
                                    torch.ones(ood_pos_con.shape[0], device=ood_pos_con.device),
                                    torch.zeros(ood_neg_con.shape[0], device=ood_neg_con.device)])

        return anchor_set, anchor_label, contrast_set, contrast_label