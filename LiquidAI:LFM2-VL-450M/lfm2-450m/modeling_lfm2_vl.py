"""PyTorch LFM2-VL model."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.lfm2.configuration_lfm2 import Lfm2Config
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2VisionModel
from transformers.processing_utils import Unpack
from transformers.utils import can_return_tuple, logging

logger = logging.get_logger(__name__)


class Lfm2VlConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Lfm2VlForConditionalGeneration`]. It is used to instantiate an
    Lfm2Vl model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Lfm2-VL-1.6B.

    e.g. [LiquidAI/LFM2-VL-1.6B](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`AutoConfig | dict`,  *optional*, defaults to `Siglip2ImageConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`AutoConfig | dict`, *optional*, defaults to `Lfm2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 396):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        projector_hidden_size (`int`, *optional*, defaults to 2056):
            The hidden size of the multimodal projector.
        projector_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the multimodal projector.
        downsample_factor (`int`, *optional*, defaults to 2):
            The downsample_factor factor of the vision backbone.
        vision_feature_layer (`int`, *optional*, defaults to -1):
            The layer of the vision tower to use as features.
        min_image_tokens (`int`, *optional*, defaults to 64):
            The minimum number of image tokens for smart resize.
        max_image_tokens (`int`, *optional*, defaults to 256):
            The maximum number of image tokens for smart resize.
        encoder_patch_size (`int`, *optional*, defaults to 16):
            The patch size of the encoder.
        max_num_patches (`int`, *optional*, defaults to 1024):
            The maximum number of image tokens passed to the encoder per image or tile.
        use_image_special_tokens (`bool`, *optional*, defaults to `True`):
            Whether to use image special tokens.
        do_image_splitting (`bool`, *optional*, defaults to `True`):
            Whether to split large images into tiles.
        min_tiles (`int`, *optional*, defaults to 2):
            The minimum number of tiles to split the image into.
        max_tiles (`int`, *optional*, defaults to 10):
            The maximum number of tiles to split the image into.
        tile_size (`int`, *optional*, defaults to 512):
            The size of the tile to split the image into.
        max_pixels_tolerance (`float`, *optional*, defaults to 2.0):
            The maximum tolerance for the number of pixels in the image before splitting.
        use_thumbnail (`bool`, *optional*, defaults to `True`):
            Whether to append the thumbnail of the image when splitting.
    """

    model_type = "lfm2-vl"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=396,
        projector_hidden_act="gelu",
        projector_hidden_size=2560,
        projector_bias=True,
        downsample_factor=2,
        vision_feature_layer=-1,
        min_image_tokens=64,
        max_image_tokens=256,
        encoder_patch_size=16,
        max_num_patches=1024,
        use_image_special_tokens=True,
        do_image_splitting=True,
        min_tiles=2,
        max_tiles=10,
        tile_size=512,
        max_pixels_tolerance=2.0,
        use_thumbnail=True,
        torch_dtype=torch.bfloat16,
        **kwargs,
    ):
        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.projector_hidden_size = projector_hidden_size
        self.projector_bias = projector_bias
        self.downsample_factor = downsample_factor
        self.vision_feature_layer = vision_feature_layer
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.encoder_patch_size = encoder_patch_size
        self.max_num_patches = max_num_patches
        self.use_image_special_tokens = use_image_special_tokens
        self.do_image_splitting = do_image_splitting
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.tile_size = tile_size
        self.max_pixels_tolerance = max_pixels_tolerance
        self.use_thumbnail = use_thumbnail
        self.torch_dtype = torch_dtype
        
        if isinstance(vision_config, dict):
            vision_config = Siglip2VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Siglip2VisionConfig()
        self.vision_config = vision_config

        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = Lfm2Config(**text_config)
        elif text_config is None:
            text_config = Lfm2Config()

        self.text_config = text_config

        super().__init__(**kwargs)


@dataclass
class Lfm2VlModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: torch.FloatTensor | None = None


@dataclass
class Lfm2VlCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class Lfm2VlMultiModalProjector(nn.Module):
    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.layer_norm = nn.LayerNorm(in_channels)
        self.linear_1 = nn.Linear(
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, image_features):
        image_features = self.layer_norm(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PixelUnshuffleBlock(nn.Module):
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, w, h, c = x.size()
        if w % self.factor != 0:
            x = torch.concat(
                [
                    x,
                    torch.zeros(
                        (n, self.factor - (w % self.factor), h, c), dtype=x.dtype
                    ).to(x.device),
                ],
                dim=1,
            ).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        if h % self.factor != 0:
            x = torch.concat(
                [
                    x,
                    torch.zeros(
                        (n, w, self.factor - (h % self.factor), c), dtype=x.dtype
                    ).to(x.device),
                ],
                dim=2,
            ).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / self.factor), int(c * self.factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n, int(h / self.factor), int(w / self.factor), int(c * self.factor**2)
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class Lfm2VlPreTrainedModel(PreTrainedModel):
    config: Lfm2VlConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]

    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = False
    _supports_flex_attn = True
    _supports_attention_backend = True


class Lfm2VlModel(Lfm2VlPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: Lfm2VlConfig):
        super().__init__(config)
        self.vision_tower = Siglip2VisionModel(config.vision_config)

        if config.vision_feature_layer != -1:
            self.vision_tower.vision_model.encoder.layers = (
                self.vision_tower.vision_model.encoder.layers[
                    : config.vision_feature_layer + 1
                ]
            )
        if config.downsample_factor > 1:
            self.pixel_unshuffle = PixelUnshuffleBlock(config.downsample_factor)
        else:
            self.pixel_unshuffle = nn.Identity()

        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        **kwargs,
    ) -> list[torch.Tensor]:
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            spatial_shapes (`torch.Tensor` of shape `(batch_size, 2)`):
                The spatial shapes of the input images.
            pixel_attention_mask (`torch.Tensor` of shape `(batch_size, height, width)`):
                The pixel attention mask of the input images.
        Returns:
            image_features (`list[torch.Tensor]`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        image_outputs = self.vision_tower(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
        ).last_hidden_state

        img_feature_lengths = pixel_attention_mask.sum(dim=1)
        image_features = []

        for img_idx in range(image_outputs.size(0)):
            feature = image_outputs[img_idx]
            # unpad the image representation
            feature = feature[: img_feature_lengths[img_idx], :].unsqueeze(0)

            feature_org_h, feature_org_w = spatial_shapes[img_idx]
            feature = feature.reshape(1, feature_org_h, feature_org_w, -1)
            feature = self.pixel_unshuffle(feature)

            # project the image representation
            img_embedding = self.multi_modal_projector(feature)

            # flatten here to handle variable length in naflex
            img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))
            image_features.append(img_embedding)

        return image_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor,
    ):
        """
        Obtains multimodal placeholdr mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        n_image_features = image_features.shape[0]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor = None,
        spatial_shapes: torch.Tensor = None,
        pixel_attention_mask: torch.Tensor = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        image_sizes: torch.Tensor = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | Lfm2VlModelOutputWithPast:
        """
        spatial_shapes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*):
            The spatial shapes of the input images.
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*):
            The pixel attention mask of the input images.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                spatial_shapes=spatial_shapes,
                pixel_attention_mask=pixel_attention_mask,
            )
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = self.get_placeholder_mask(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return Lfm2VlModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class Lfm2VlForConditionalGeneration(Lfm2VlPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Lfm2VlConfig):
        super().__init__(config)
        self.model = Lfm2VlModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def _supports_default_dynamic_cache(self):
        return False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        spatial_shapes: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        **kwargs,
    ):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            **kwargs,
        )

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        spatial_shapes: torch.Tensor = None,
        pixel_attention_mask: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_sizes: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | Lfm2VlCausalLMOutputWithPast:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`, *optional*):
            The input image tensors.
        spatial_shapes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*):
            The spatial shapes of the input images.
        pixel_attention_mask (`torch.Tensor` of shape `(batch_size, height, width)`, *optional*):
            The pixel attention mask of the input images.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModelForImageTextToText
        >>> from transformers.image_utils import load_image

        >>> model = AutoModelForImageTextToText.from_pretrained(
        ...     "LiquidAI/LFM2-VL-1.6B",
        ...     trust_remote_code=True
        ... )
        >>> processor = AutoProcessor.from_pretrained(
        ...     "LiquidAI/LFM2-VL-1.6B",
        ...     trust_remote_code=True
        ... )

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = load_image(url)

        >>> conversation = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": image},
        ...             {"type": "text", "text": "What is in this image?"},
        ...         ],
        ...     },
        ... ]

        >>> inputs = processor.apply_chat_template(
        ...     conversation,
        ...     add_generation_prompt=True,
        ...     tokenize=True,
        ...     return_dict=True,
        ...     return_tensors="pt"
        ... )

        >>> # Generate
        >>> outputs = model.generate(**inputs, max_new_tokens=45)
        >>> processor.batch_decode(outputs, skip_special_tokens=True)[0]
        'This image depicts a vibrant street scene in what appears to be a Chinatown or similar cultural area. The focal point is a large red stop sign with white lettering, mounted on a pole.'
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            spatial_shapes=spatial_shapes,
            pixel_attention_mask=pixel_attention_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **kwargs,
            )

        return Lfm2VlCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values

        return model_inputs


__all__ = ["Lfm2VlForConditionalGeneration", "Lfm2VlModel", "Lfm2VlPreTrainedModel"]
