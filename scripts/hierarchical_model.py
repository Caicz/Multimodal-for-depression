import torch
import torch.nn as nn
import math
from transformers import BertModel
from torchvision.models import resnet18, ResNet18_Weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

VLM_bert_model_name = 'hfl/chinese-roberta-wwm-ext'

class HierarchicalMultimodalModel(nn.Module):
    def __init__(self, num_classes=2, seq_len=25, unfreeze_bert=True, unfreeze_resnet=True, bert_model_name='distilbert-base-uncased', ablation=None):
        super(HierarchicalMultimodalModel, self).__init__()
        self.seq_len = seq_len
        self.unfreeze_bert = unfreeze_bert
        self.unfreeze_resnet = unfreeze_resnet  
        self.ablation = ablation
        # 基础特征提取
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.vlm_bert = BertModel.from_pretrained(VLM_bert_model_name)
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(self.resnet.children())[:-2]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        #消融实验设置：根据 ablation 参数控制不同模块的启用/禁用
        if ablation == 'textonly':
            self.use_img = False
            self.use_crossattn = False
            self.use_rt = False
            self.use_density = False
        elif ablation == 'img':
            self.use_img = False
            self.use_crossattn = False
            self.use_rt = True
            self.use_density = True
        elif ablation == 'crossattn':
            self.use_img = True
            self.use_crossattn = False
            self.use_rt = True
            self.use_density = True
        elif ablation == 'rt':
            self.use_img = True
            self.use_crossattn = True
            self.use_rt = False
            self.use_density = True
        elif ablation == 'density':
            self.use_img = True
            self.use_crossattn = True
            self.use_rt = True
            self.use_density = False
        else:  # None 或 'full'
            self.use_img = True
            self.use_crossattn = True
            self.use_rt = True
            self.use_density = True
        
        # 特征投影层
        self.text_projection = nn.Linear(768, 112)
        self.img_projection = nn.Linear(512, 112)
        self.summary_projection = nn.Linear(768, 128)   # 降维：将 VLM 总结作为辅助特征
        self.rt_projection = nn.Linear(1, 32)
        self.density_projection = nn.Linear(1, 16)     # 降维至 16，作为轻量辅助

        # 模态权重 (可学习的缩放因子)
        self.modal_weights = nn.Parameter(torch.ones(3)) # [序列权重, VLM权重, 密度权重]

        # 模态内交互 (Cross-Attention)
        if self.use_img and self.use_crossattn:
            self.cross_attn = nn.MultiheadAttention(embed_dim=112, num_heads=4, batch_first=True)
            self.attn_norm = nn.LayerNorm(112)
        else:
            self.cross_attn = None
            self.attn_norm = None

        tweet_dim = 112  # 文本总是存在
        if self.use_img:
            tweet_dim += 112
        if self.use_rt:
            tweet_dim += 32
        self.d_model = tweet_dim

        # 序列建模 (Transformer)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # 6. 分类器
        fusion_dim = self.d_model + 128
        if self.use_density:
            fusion_dim += 16
        self.final_norm = nn.LayerNorm(fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, num_classes),
            nn.Dropout(0.5)
        )
        # 冻结/解冻策略
        for param in self.bert.parameters(): param.requires_grad = False
        for param in self.resnet_backbone.parameters(): param.requires_grad = False
        for param in self.vlm_bert.parameters(): param.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values, seq_mask, risk_density, is_retweet, summary_ids=None, summary_mask=None, return_attn=False):
        B, S, T = input_ids.shape
        flat_input_ids = input_ids.view(B * S, T)
        flat_attn_mask = attention_mask.view(B * S, T)
        flat_pixels = pixel_values.view(B * S, 3, 224, 224)
        
        # --- 分支 1: 推文序列 (256维) ---
        text_out = self.bert(flat_input_ids, attention_mask=flat_attn_mask).last_hidden_state[:, 0, :]
        text_feat = self.text_projection(text_out)
        
        img_feat = None
        cross_attn_weights = None

        # 图像特征提取和交叉注意力
        if self.use_img:
            img_features = self.resnet_backbone(flat_pixels)
            img_global = self.avgpool(img_features).view(B*S, 512)
            img_feat_raw = self.img_projection(img_global)   # [B*S, 112]

            if self.use_crossattn:
                # 交叉注意力，返回注意力权重
                img_context, cross_attn_weights = self.cross_attn(
                    text_feat.unsqueeze(1), img_feat_raw.unsqueeze(1), img_feat_raw.unsqueeze(1))
                img_feat = self.attn_norm(img_feat_raw + img_context.squeeze(1))
            else:
                # 无交叉注意力，直接使用原始投影
                img_feat = img_feat_raw
        
        rt_feat = self.rt_projection(is_retweet.view(B * S, 1).float())
        
        # 拼接动态特征
        features = [text_feat]
        if img_feat is not None:
            features.append(img_feat)
        if self.use_rt:
            rt_feat = self.rt_projection(is_retweet.view(B * S, 1).float())
            features.append(rt_feat)

        tweet_feat = torch.cat(features, dim=-1) 
        seq_feat = tweet_feat.view(B, S, -1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        combined_seq = torch.cat([cls_tokens, seq_feat], dim=1)
        combined_seq = self.pos_encoder(combined_seq)
        
        cls_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids.device)
        combined_mask = torch.cat([cls_mask, seq_mask], dim=1)
        # 手动遍历 Transformer 层以提取注意力权重
        transformer_attn_weights = []
        x = combined_seq
        src_key_padding_mask = ~combined_mask
        
        for layer in self.transformer.layers:
            # 1. Self-Attention
            residual = x
            if layer.norm_first:
                nx, attn = layer.self_attn(layer.norm1(x), layer.norm1(x), layer.norm1(x), key_padding_mask=src_key_padding_mask)
                x = residual + layer.dropout1(nx)
                # 2. Feed-Forward
                residual = x
                x = residual + layer._ff_block(layer.norm2(x))
            else:
                nx, attn = layer.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
                x = layer.norm1(residual + layer.dropout1(nx))
                # 2. Feed-Forward
                x = layer.norm2(x + layer._ff_block(x))
            transformer_attn_weights.append(attn)

        pooled_seq_feat = x[:, 0, :] # [B, 256]
        
        # --- 分支 2: VLM 总结 (128维) ---
        if summary_ids is not None and summary_mask is not None:
            vlm_output = self.vlm_bert(summary_ids, attention_mask=summary_mask).last_hidden_state
            # 对所有非 Padding 的 Token 取平均
            sum_mask_expanded = summary_mask.unsqueeze(-1).expand(vlm_output.size()).float()
            sum_out = torch.sum(vlm_output * sum_mask_expanded, 1) / torch.clamp(sum_mask_expanded.sum(1), min=1e-9)
            user_sum_feat = self.summary_projection(sum_out)
        else:
            user_sum_feat = torch.zeros(B, 128).to(input_ids.device)
            
        # --- 分支 3: 风险密度 (16维) ---
        if self.use_density:
            user_density_feat = self.density_projection(risk_density.view(B, 1))   # [B, 16]
        else:
            user_density_feat = None
        
        # --- 加权融合 (256 + 128 + 16 = 400) ---
        weighted_seq = pooled_seq_feat * self.modal_weights[0]
        weighted_sum = user_sum_feat * self.modal_weights[1]

        to_concat = [weighted_seq, weighted_sum]
        if user_density_feat is not None:
            weighted_density = user_density_feat * self.modal_weights[2]
            to_concat.append(weighted_density)

        final_feat = torch.cat(to_concat, dim=-1)
        final_feat = self.final_norm(final_feat)
        
        logits = self.classifier(final_feat)
        
        if return_attn:
            return logits, transformer_attn_weights, cross_attn_weights
        return logits
