from __future__ import annotations

import tempfile
from groundingdino.models.GroundingDINO import groundingdino
from groundingdino.models.GroundingDINO.groundingdino import *
from groundingdino.util.inference import *
from typing import *

from elsa import util
from tools.inference_on_a_image import *


# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.

class LocalFiles(util.LocalFiles):
    # To achieve this, I used
    # echo 'export elsa="dhodcz2"' >> venv/bin/activate && source venv/bin/activate
    config: str = dict(
        dhodcz2='/home/arstneio/PycharmProjects/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py',
        marco="/home/mcipriano/projects/elsa/configs/cfg_odvg.py",
        marco_KIZ="/home/marco.cipriano/projects/elsa/configs/cfg_odvg.py"
    )
    checkpoint: str = dict(
        dhodcz2='/home/arstneio/Downloads/gdinot-coco-ft.pth',
        marco="/home/mcipriano/projects/elsa/weights/gdinot-coco-ft.pth",
        marco_KIZ="/home/marco.cipriano/projects/elsa/weights/groundingdino_swint_ogc.pth"
    )

    batch_size: int = dict(
        dhodcz2=32,
        marco=64,
        marco_KIZ=128
    )

class Result(NamedTuple):
    pred_logits: torch.Tensor
    pred_boxes: torch.Tensor
    input_ids: torch.Tensor
    offset_mapping: torch.Tensor


class GroundingDINO(groundingdino.GroundingDINO):
    @classmethod
    def from_elsa(
            cls,
            config: str = None,
            checkpoint: str = None,
    ) -> Self:
        if config is None:
            config = LocalFiles.config
        if checkpoint is None:
            checkpoint = LocalFiles.checkpoint
        args = SLConfig.fromfile(config)
        args.device = 'cuda'
        # model = build_model(args)
        assert 'elsa_groundingdino' in MODULE_BUILD_FUNCS._module_dict
        build_func = MODULE_BUILD_FUNCS.get('elsa_groundingdino')
        model = build_func(args)
        checkpoint = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        return model

    def forward(
            self,
            samples: NestedTensor,
            targets: List = None,
            **kw
    ) -> Result:
        """
        The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]

        tokenized = self.tokenizer(
            captions,
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(samples.device)
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {
                k: v
                for k, v in tokenized.items()
                if k not in ["attention_mask", "offset_mapping"]
            }
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                                        :, : self.max_text_len, : self.max_text_len
                                        ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # output
        layers = [
            embed(hs, text_dict)
            for embed, hs in zip(self.class_embed, hs)
        ]
        outputs_class = torch.stack(layers)
        out = Result(
            pred_logits=outputs_class[-1],
            pred_boxes=outputs_coord_list[-1],
            input_ids=tokenized.input_ids,
            offset_mapping=tokenized.offset_mapping,
        )

        return out


@MODULE_BUILD_FUNCS.registe_with_name(module_name='elsa_groundingdino')
def build_groundingdino(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )

    return model


def inference_on_an_image(
        image_path: str,
        text_prompt: str,
        token_spans: Optional[List[Tuple[int, int]]] = None,
        box_threshold: float = .3,
        text_threshold: float = .25,
        config_file: str = None,
        checkpoint_path: str = None,
):
    if config_file is None:
        config_file = LocalFiles.config
    if checkpoint_path is None:
        checkpoint_path = LocalFiles.checkpoint
    cpu_only = False
    output_dir = tempfile.gettempdir()

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    # model = load_model(config_file, checkpoint_path, cpu_only=False)
    model = GroundingDINO.from_elsa(config_file, checkpoint_path)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # set the text_threshold to None if token_spans is set.
    if token_spans is not None:
        text_threshold = None
        print("Using token_spans. Set the text_threshold to None.")

    # run model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, token_spans=token_spans
    )

    # visualize pred
    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    save_path = os.path.join(output_dir, "pred.jpg")
    image_with_box.save(save_path)
    print(f"Saved the result to {save_path}")


if __name__ == '__main__':
    # image_path = '/home/arstneio/Downloads/Archive/bing/020310023333001110_x4_cropped.png'
    image_path = '/home/arstneio/Downloads/Archive/bing/103330322123201010_x4_cropped.png'
    # prompt = 'a person riding a bike'
    prompt = 'a group standing to cross a crosswalk'
    inference_on_an_image(image_path, prompt)
