# coding=utf-8
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch XLM-RoBERTa model."""
from typing import List, Optional, Tuple, Union
import torch
from torch import nn

from ...activations import ACT2FN, gelu
from ...utils import add_start_docstrings, logging
from ..roberta.modeling_roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaEmbeddings,
    RobertaPooler,
    RobertaLayer,
)
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from .configuration_meae import XLMRobertaConfig


logger = logging.get_logger(__name__)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlm-roberta-large-finetuned-conll02-dutch",
    "xlm-roberta-large-finetuned-conll02-spanish",
    "xlm-roberta-large-finetuned-conll03-english",
    "xlm-roberta-large-finetuned-conll03-german",
    # See all XLM-RoBERTa models at https://huggingface.co/models?filter=xlm-roberta
]


XLM_ROBERTA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XLMRobertaConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaModel(RobertaModel):
    """
    This class overrides [`RobertaModel`]. Please check the superclass for the appropriate documentation alongside
    usage examples.
    """

    config_class = XLMRobertaConfig


class MEAEModel(RobertaPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = MEAEEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class MEAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        # for MEAE
        if config.entity_memory_after_layers:
            self.add_entity_memory_after_layers = config.add_entity_memory_after_layers
            #[int(i) for i in config.entity_memory_after_layers.split(",")]
        else:
            self.add_entity_memory_after_layers = []
        logger.info("Adding entity memory after transformer layer: {}".format(self.add_entity_memory_after_layers))

        self.entity_vocab_size = config.entity_vocab_size
        self.entity_embed_dim = config.entity_embed_dim
        self.entity_padding_idx = config.entity_padding_idx

        self.entity_memory = EntityMemory(
            encoder_embed_dim = config.hidden_size,
            entity_vocab_size = self.entity_vocab_size,
            entity_embed_dim = self.entity_embed_dim,
            mention_to_add_entity_embedding = config.mention_to_add_entity_embedding)
        self.entity_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.bio_head = BioClassificationHead(config, num_labels=3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class EntityMemory(nn.Module):
    """
    Entity Memory, as described in the paper
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        entity_vocab_size: int,
        entity_embed_dim: int,
        mention_to_add_entity_embedding: str = "first",
    ):
        """
        :param encoder_embed_dim the size of an embedding. In the EaE paper it is called d_emb, previously as d_k
            (attention_heads * embedding_per_head)
        :param entity_vocab_size also known as N in the EaE paper, the maximum number of entities we store
        :param entity_embed_dim also known as d_ent in the EaE paper, the embedding of each entity
        :param mention_to_add_entity_embedding where to add the entity embedding to (first/last/all mention tokens)
        :param freeze if true, freeze this module. This enforces k-nearest fetching all the time

        """
        super().__init__()
        self.mention_to_add_entity_embedding = mention_to_add_entity_embedding
        logger.info("Adding weighted entity embeddings to {} mention token(s)".format(mention_to_add_entity_embedding))
        # pylint:disable=invalid-name
        self.entity_vocab_size = entity_vocab_size
        self.encoder_embed_dim = encoder_embed_dim
        self.entity_embed_dim = entity_embed_dim
        # pylint:disable=invalid-name
        self.hidden_to_entity = nn.Linear(2*encoder_embed_dim, self.entity_embed_dim)
        # pylint:disable=invalid-name
        self.entity_to_hidden = nn.Linear(self.entity_embed_dim, encoder_embed_dim)
        # pylint:disable=invalid-name
        self.entity_embedding = nn.Linear(self.entity_vocab_size, self.entity_embed_dim, bias=False)

        #print ("self.entity_embedding:", self.entity_embedding.weight.shape)
        # TODO: Do not make these hardcoded.
        # The BIO class used to hold these but it got deprecated...
        self.begin = 1
        self.inner = 2
        self.out = 0

        self.loss = nn.NLLLoss()

    def _get_last_mention(self, bio_output, pos):
        end_mention = pos[1]

        for end_mention in range(pos[1] + 1, bio_output.size(1)):
            if bio_output[pos[0], end_mention] != self.inner:
                break
        end_mention -= 1

        return end_mention

    def _get_k_nearest(
        self,
        pseudo_entity_embedding: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        # K nearest neighbours
        # Note: the paper makes a slight notation abuse.
        # When computing the query vector, alpha is the softmax of the topk entities
        # When computing the loss, alpha is the softmax across the whole dictionary
        topk = torch.topk(pseudo_entity_embedding @ self.entity_embedding.weight, k, dim=1)

        #alpha_topk = F.softmax(topk.values, dim=1)
        alpha_topk = nn.functional.softmax(topk.values, dim=1)

        # mat1 has size (M x d_ent x k), mat2 has size (M x k x 1)
        # the result has size (M x d_ent x 1). Squeeze that out and we've got our
        # entities of size (M x d_ent).
        picked_entity = torch.bmm(
            self.entity_embedding.weight[:, topk.indices].transpose(0, 1),
            alpha_topk.view((-1, k, 1))).view((-1, self.entity_embed_dim))

        return picked_entity


    def forward(
        self,
        X,
        bio_output: torch.Tensor,
        entities_output: Optional[torch.Tensor],
        k=100,
        debug=False,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        :param x the (raw) output of the first transformer block. It has a shape:
                B x N x (embed_size).
        :param bio_output the output of the bio classifier. If not provided no loss is returned
                (which is required during a training stage).
        :param entities_output the detected entities. If not provided no loss is returned
                (which is required during a training stage).
        :param k
        :returns a pair (loss, transformer_output). If either of entities_output or bio_output is
                  None loss will be None as well.
        """

        is_supervised = entities_output is not None
        assert not self.training or is_supervised is not None, \
            "Cannot perform training without entities_output"

        DEVICE = X.device

        weighted_entity_embedding = torch.zeros_like(X).to(DEVICE)

        # Disable gradient calculation for BIO outputs, but re-enable them
        # for the span
        with torch.no_grad():
            #loss = None
            #if is_supervised:
            #	loss = torch.zeros((1,)).to(DEVICE)

            begin_positions = torch.nonzero(bio_output == self.begin)

            # if no mentions are detected skip the entity memory.
            if len(begin_positions) == 0:
                return None, None, weighted_entity_embedding

            # FIXME: Not really parallelized (we don't have vmap yet...)
            end_positions = torch.tensor([
                self._get_last_mention(bio_output, pos) for pos in begin_positions]).unsqueeze(1).to(DEVICE)


        # Create the tensor so that it contains the batch position, the begin_positions
        # and the end positions in separate rows.
        positions = torch.cat([begin_positions, end_positions], 1).T

        first = X[positions[0], positions[1]]
        second = X[positions[0], positions[2]]

        mention_span = torch.cat([first, second], 1).to(DEVICE)

        pseudo_entity_embedding = self.hidden_to_entity(
            mention_span)  # num_of_mentions x d_ent
        if debug:
            print ("mention_span:\n{}".format(mention_span))
            print ("pseudo_entity_embed:\n{}".format(pseudo_entity_embedding))

        # If supervised, ALWAYS compute the loss
        #   If supervised and training, perform knn
        # If not supervised, perform knn and do not compute the loss

        if is_supervised:
            score = pseudo_entity_embedding.matmul(self.entity_embedding.weight)
            #alpha = F.softmax(score, dim=1)
            alpha = nn.functional.softmax(score, dim=1)
            #alpha = F.softmax(
            #	pseudo_entity_embedding.matmul(self.entity_embedding.weight), dim=1)
            if debug:
                print ("score:\n{}".format(score))
                torch.set_printoptions(profile="full")
                print ("alpha:(shape:{})\n{}".format(alpha.shape, alpha[:,:100]))
                torch.set_printoptions(profile="default")

            #alpha_log = torch.log(alpha)
            # Compared to the original paper we use NLLoss.
            # Gradient-wise this should not change anything
            #print ("entities_output:\n", entities_output)
            #print ("entities_output[positions[0], positions[1]]:\n", entities_output[positions[0], positions[1]])
            #loss = self.loss(
            #	alpha_log, entities_output[positions[0], positions[1]])

            # NOTE: self.training may be on even when the memory is frozen
            if self.training:
                # shape: B x d_ent
                # k = entity_vocab_size while training
                picked_entity = self.entity_embedding(alpha)
            else:
                picked_entity = self._get_k_nearest(pseudo_entity_embedding, k)
        
        else:
            picked_entity = self._get_k_nearest(pseudo_entity_embedding, k)

        if self.mention_to_add_entity_embedding == "first":
            # (batch_size, seq_len, enc_dim)
            weighted_entity_embedding[positions[0], positions[1]] = self.entity_to_hidden(picked_entity)
        elif self.mention_to_add_entity_embedding == "last":
            weighted_entity_embedding[positions[0], positions[2]] = self.entity_to_hidden(picked_entity)
        elif self.mention_to_add_entity_embedding == "first-last":
            weighted_entity_embedding[positions[0], positions[1]] = self.entity_to_hidden(picked_entity)
            weighted_entity_embedding[positions[0], positions[2]] = self.entity_to_hidden(picked_entity)
        elif self.mention_to_add_entity_embedding == "all":
            mention_lens = positions[2]-positions[1]+1
            max_mention_len = mention_lens.max().item()
            #print ("mention_lens:\n", mention_lens)
            #print ("max_mention_len=", max_mention_len)
            for offset in range(max_mention_len):
                target_positions = torch.where(mention_lens>offset, positions[1]+offset, positions[1])
                weighted_entity_embedding[positions[0], target_positions] = self.entity_to_hidden(picked_entity)

        if debug:
            print ("positions:(type={})\n{}".format(positions.dtype, positions))
            print ("weighted_entity_embedding.nonzero():\n{}".format(weighted_entity_embedding[:,:,0].nonzero()))
            print ("weighted_entity_embedding[pos[0],pos[1]]:\n{}".format(weighted_entity_embedding[positions[0], positions[1]]))

        #return alpha, positions, weighted_entity_embedding
        # returning score instead of alpha to fix fp16 overflow
        return score, positions, weighted_entity_embedding

class BioClassificationHead(nn.Module):
    """
    BIO classifier head
    """

    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, num_labels)
        self.bias = nn.Parameter(torch.zeros(num_labels))
        self.decoder.bias = self.bias

    # do not use masked_tokens since it is not related to mention BIO
    def forward(self, features, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation

        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        # (batch, seq_len, num_labels)
        x = self.decoder(x)
        return x


class EntityPredictionHead(nn.Module):
    """
    Entity pred head, similar to EntityMemory but takes as input the top hidden states
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        entity_vocab_size: int,
        entity_embed_dim: int,
        entity_embedding,
    ):
        """
        :param encoder_embed_dim the size of an embedding. In the EaE paper it is called d_emb, previously as d_k
            (attention_heads * embedding_per_head)
        :param entity_vocab_size also known as N in the EaE paper, the maximum number of entities we store
        :param entity_embed_dim also known as d_ent in the EaE paper, the embedding of each entity
        :param mention_to_add_entity_embedding where to add the entity embedding to (first/last/all mention tokens)
        :param freeze if true, freeze this module. This enforces k-nearest fetching all the time

        """
        super().__init__()
        # pylint:disable=invalid-name
        self.entity_vocab_size = entity_vocab_size
        self.encoder_embed_dim = encoder_embed_dim
        self.entity_embed_dim = entity_embed_dim
        # pylint:disable=invalid-name
        self.hidden_to_entity = nn.Linear(2*encoder_embed_dim, self.entity_embed_dim)
        # pylint:disable=invalid-name
        self.entity_to_hidden = nn.Linear(self.entity_embed_dim, encoder_embed_dim)
        # pylint:disable=invalid-name
        self.entity_embedding = entity_embedding

        #print ("self.entity_embedding:", self.entity_embedding.weight.shape)
        # TODO: Do not make these hardcoded.
        # The BIO class used to hold these but it got deprecated...
        self.begin = 1
        self.inner = 2
        self.out = 0

        self.loss = nn.NLLLoss()

    def _get_last_mention(self, bio_output, pos):
        end_mention = pos[1]

        for end_mention in range(pos[1] + 1, bio_output.size(1)):
            if bio_output[pos[0], end_mention] != self.inner:
                break
        end_mention -= 1

        return end_mention

    def forward(
        self,
        X,
        bio_output: torch.Tensor,
        entities_output: Optional[torch.Tensor],
        positions = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        :param x the (raw) output of the first transformer block. It has a shape:
                B x N x (embed_size).
        :param bio_output the output of the bio classifier. If not provided no loss is returned
                (which is required during a training stage).
        :param entities_output the detected entities. If not provided no loss is returned
                (which is required during a training stage).
        :param positions the positions computed in EntityMemory, no need to compute it again if provided
        
        :returns alpha. The weights of each entity.
        """

        DEVICE = X.device

        weighted_entity_embedding = torch.zeros_like(X).to(DEVICE)

        # if positions not provided, compute it here
        if positions is None:
            # Disable gradient calculation for BIO outputs, but re-enable them
            # for the span
            with torch.no_grad():
                begin_positions = torch.nonzero(bio_output == self.begin)

                # if no mentions are detected skip the entity memory.
                if len(begin_positions) == 0:
                    return torch.zeros((1,)).to(DEVICE), weighted_entity_embedding

                # FIXME: Not really parallelized (we don't have vmap yet...)
                end_positions = torch.tensor([
                    self._get_last_mention(bio_output, pos) for pos in begin_positions]).unsqueeze(1).to(DEVICE)


            # Create the tensor so that it contains the batch position, the begin_positions
            # and the end positions in separate rows.
            positions = torch.cat([begin_positions, end_positions], 1).T

        first = X[positions[0], positions[1]]
        second = X[positions[0], positions[2]]

        mention_span = torch.cat([first, second], 1).to(DEVICE)

        pseudo_entity_embedding = self.hidden_to_entity(
            mention_span)  # num_of_mentions x d_ent

        score = pseudo_entity_embedding.matmul(self.entity_embedding.weight)
        #alpha = F.softmax(score, dim=1)
        alpha = nn.functional.softmax(score, dim=1)



@add_start_docstrings(
    "XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForCausalLM(RobertaForCausalLM):
    """
    This class overrides [`RobertaForCausalLM`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """XLM-RoBERTa Model with a `language modeling` head on top.""",
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides [`RobertaForMaskedLM`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides [`RobertaForMultipleChoice`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides [`RobertaForTokenClassification`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = XLMRobertaConfig


@add_start_docstrings(
    """
    XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides [`RobertaForQuestionAnswering`]. Please check the superclass for the appropriate documentation
    alongside usage examples.
    """

    config_class = XLMRobertaConfig
