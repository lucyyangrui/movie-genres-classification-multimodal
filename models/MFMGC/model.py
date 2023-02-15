import torch
import torch.nn as nn
from transformers import PretrainedConfig, SwinConfig, SwinModel
from transformers.models.bert.modeling_bert import BertConfig, BertPreTrainedModel, \
        BertEmbeddings, BertEncoder, BertModel

import sys
from transformers import Wav2Vec2Model

sys.path.append('/opt/data/private/research/experiment/models')

from swin.swin import swin_small


class MFMGP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modals = args.modals
        self.device = args.device
        self.cls_weight = args.cls_weight
        self.bert_cfg = BertConfig.from_pretrained(args.bert_dir)
        self.hidden_size = self.bert_cfg.hidden_size
        self.audio_hidden_size = args.wav2vec_hidden_size

        self.visual_backbone = swin_small(args.swin_pretrained_path)
        self.bert = VlBertModel.from_pretrained(args.bert_dir)
        self.visual_linear = nn.Linear(self.hidden_size, self.hidden_size)

        # self.audio_backbone = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
        # self.audio_backbone.feature_extractor.requires_grad_(False)  # not training
        # self.audio_backbone.feature_projection.requires_grad_(False)
        # self.audio_backbone.encoder.requires_grad_(False)
        self.audio_linear = nn.Linear(self.audio_hidden_size, self.hidden_size)

        self.out_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, args.cls_num)
        
        self.dropout = nn.Dropout(0.5)

    def encoder_frames(self, visual_input):
        assert len(visual_input.size()) == 5, print('visual input size must equal to 5!')
        bs = visual_input.size(0)
        frames = visual_input.size(1)
        visual_input = visual_input.view((bs * frames,) + visual_input.size()[2:])
        encoder_output = self.visual_backbone(visual_input)
        encoder_output = encoder_output.view(bs, frames, encoder_output.size(-1))
        return encoder_output

    # def encode_audio(self, audio_list):
        # assert len(audio_list.size()) == 3, 'audio list size must equal to 3!'
        # bs = audio_list.size(0)
        # partial_num = audio_list.size(1)
        # audio_list = audio_list.view((bs * partial_num, audio_list.size(-1)))  # [bs * n, sr * partial_len]
        # audio_feature = self.audio_backbone(audio_list).last_hidden_state  # [ba * n, 149, 768]
        # audio_feature = torch.mean(audio_feature, dim=1)  # [bs * n, 768]
        # audio_feature = audio_feature.view((bs, partial_num, audio_feature.size(-1)))  # [bs, n, 768]
        # return audio_feature

    def cal_loss(self, prediction, label):
        # label --> [bs, 25(cls_num)]
        assert prediction.size() == label.size(), 'prediction size must equal to label'
        label_float = label.float()
        loss = nn.BCEWithLogitsLoss(pos_weight=self.cls_weight.to(self.device))  # (pos_weight=cls_weight.cuda())
        loss = loss(prediction, label_float)
        with torch.no_grad():
            prediction_sig = torch.sigmoid(prediction)
            pred_label_id = torch.where(prediction_sig > 0.5,
                                        torch.ones_like(label), torch.zeros_like(label))  # [bs, 25]  0 1 组成
            true_label_num = torch.sum(pred_label_id == label)
            # if loss.mean() > 2.0:
                # print('prediction: ', prediction)
                # print('label_pro: ', pred_label_id)
                # print('true_label: ', label)
            accuracy = true_label_num / (label.size(0) * label.size(1))
        return loss, accuracy, pred_label_id

    def forward(self, data, get_node_feat=False):
        inputs = {}
        inputs_mask = {}
        if 'summary' in self.modals:
            text_input = data['text_input']  # [bs, seq_len]
            text_mask = data['text_mask']  # [bs, seq_len]
            inputs['text_input'] = text_input
            inputs_mask['text_mask'] = text_mask

        if 'video' in self.modals:
            frame_input = data['frame_input']  # [bs, 32, 3, 224, 224]
            frame_mask = data['frame_mask']  # [bs, 32]
            visual_feature = self.encoder_frames(frame_input)  # [bs, 32, 768]
            inputs['visual_feature'] = visual_feature
            inputs_mask['visual_mask'] = frame_mask

        if 'audio' in self.modals:
            audio_feature = data['audio_wav_feat']  # [bs, n, 1024]
            audio_mask = data['audio_mask']
            audio_feature = self.audio_linear(audio_feature) # [bs, n, 768]
            inputs['audio_feature'] = audio_feature
            inputs_mask['audio_mask'] = audio_mask

        output = self.bert(inputs, inputs_mask)  # [bs, seq_len + 32, 768]

        pooled_output = torch.mean(output, dim=1)  # [bs, 768]

        # pooled_output = self.out_linear(pooled_output)
        if get_node_feat:
            return pooled_output
            
        classifier = self.classifier(pooled_output)  # [bs, 10]

        loss, acc, pred_label_id = self.cal_loss(classifier, data['label'])
        return loss, acc, pred_label_id, classifier


class VlBertModel(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Module):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, inputs: dict, inputs_mask: dict): # ):
        encoder_input = []
        encoder_mask = []
        if 'text_input' in inputs:
            text_input = self.embeddings(inputs['text_input'])
            encoder_input.append(text_input)
            encoder_mask.append(inputs_mask['text_mask'])
        if 'visual_feature' in inputs:
            visual_emb = self.embeddings(inputs_embeds=inputs['visual_feature'], token_type_ids=torch.ones_like(inputs_mask['visual_mask']))
            encoder_input.append(visual_emb)
            encoder_mask.append(inputs_mask['visual_mask'])
        if 'audio_feature' in inputs:
            audio_emb = self.embeddings(inputs_embeds=inputs['audio_feature'], token_type_ids=torch.ones_like(inputs_mask['audio_mask']))
            encoder_input.append(audio_emb)
            encoder_mask.append(inputs_mask['audio_mask'])

        embeddings = torch.cat(encoder_input, dim=1) # 
        mask = torch.cat(encoder_mask, dim=1) # 
        # embeddings = text_input
        # mask = text_mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0

        encoder_output = self.encoder(embeddings, attention_mask=mask)['last_hidden_state']
        return encoder_output

