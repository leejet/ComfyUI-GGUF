# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
import warnings
import logging
import torch
import gguf
import re
import os

from .ops import GGMLTensor
from .dequant import is_quantized, dequantize_tensor

IMG_ARCH_LIST = {"flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2",
                 "qwen_image", "ideogram4", "krea2", "minimax", "qwen_image21"}
TXT_ARCH_LIST = {"t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "qwen35", "gemma3"}
VIS_TYPE_LIST = {"clip-vision", "mmproj"}

def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        # extra check here as this is used for checking arch string
        if len(field.types) != 1 or field.types[0] != gguf.GGUFValueType.STRING:
            raise TypeError(f"Bad type for GGUF {field_name} key: expected string, got {field.types!r}")
        return str(field.parts[field.data[-1]], encoding="utf-8")
    elif field_type in [int, float, bool]:
        return field_type(field.parts[field.data[-1]].item())
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_list_field(reader, field_name, field_type):
    field = reader.get_field(field_name)
    if field is None:
        return None
    elif field_type == str:
        return tuple(str(field.parts[part_idx], encoding="utf-8") for part_idx in field.data)
    elif field_type in [int, float, bool]:
        return tuple(field_type(field.parts[part_idx][0]) for part_idx in field.data)
    else:
        raise TypeError(f"Unknown field type {field_type}")

def get_gguf_metadata(reader):
    """Extract all simple metadata fields like safetensors"""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:  # Simple scalar fields only
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] in (gguf.GGUFValueType.INT32, gguf.GGUFValueType.UINT32):
                    # UINT32 parts arrive as 1-element arrays, so unwrap before casting.
                    metadata[field_name] = int(field.parts[field.data[-1]].reshape(-1)[0])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except:
            continue
    return metadata

def gguf_sd_loader(path, handle_prefix="model.diffusion_model.", is_text_model=False):
    """
    Read state dict as fake tensors
    """
    reader = gguf.GGUFReader(path)

    # filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # detect and verify architecture
    compat = None
    arch_str = get_field(reader, "general.architecture", str)
    type_str = get_field(reader, "general.type", str)
    if arch_str in [None, "pig", "cow"]:
        if is_text_model:
            raise ValueError(f"This gguf file is incompatible with llama.cpp!\nConsider using safetensors or a compatible gguf file\n({path})")
        compat = "sd.cpp" if arch_str is None else arch_str
        # import here to avoid changes to convert.py breaking regular models
        from .tools.convert import detect_arch
        try:
            arch_str = detect_arch(set(val[0] for val in tensors)).arch
        except Exception as e:
            raise ValueError(f"This model is not currently supported - ({e})")
    elif arch_str not in TXT_ARCH_LIST and is_text_model:
        if type_str not in VIS_TYPE_LIST:
            raise ValueError(f"Unexpected text model architecture type in GGUF file: {arch_str!r}")
    elif arch_str not in IMG_ARCH_LIST and not is_text_model:
        raise ValueError(f"Unexpected architecture type in GGUF file: {arch_str!r}")

    if compat:
        logging.warning(f"Warning: This gguf model file is loaded in compatibility mode '{compat}' [arch:{arch_str}]")

    # main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        # torch_tensor = torch.from_numpy(tensor.data) # mmap

        # NOTE: line above replaced with this block to avoid persistent numpy warning about mmap
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data) # mmap

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            # Workaround for stable-diffusion.cpp SDXL detection.
            if compat == "sd.cpp" and arch_str == "sdxl":
                if any([tensor_name.endswith(x) for x in (".proj_in.weight", ".proj_out.weight")]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]

        # add to state dict
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # 1D tensors shouldn't be quantized, this is a fix for BF16 and for
        # sd.cpp builds that quantize 1D norm weights
        if len(shape) <= 1 and is_quantized(state_dict[sd_key]):
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        # keep track of loaded tensor types
        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    # print loaded tensor type counts
    logging.info("gguf qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # mark largest tensor for vram estimation
    qsd = {k:v for k,v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    # extra info to return
    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader)
    }
    return (state_dict, extra)

# for remapping llama.cpp -> original key names
T5_SD_MAP = {
    "enc.": "encoder.",
    ".blk.": ".block.",
    "token_embd": "shared",
    "output_norm": "final_layer_norm",
    "attn_q": "layer.0.SelfAttention.q",
    "attn_k": "layer.0.SelfAttention.k",
    "attn_v": "layer.0.SelfAttention.v",
    "attn_o": "layer.0.SelfAttention.o",
    "attn_norm": "layer.0.layer_norm",
    "attn_rel_b": "layer.0.SelfAttention.relative_attention_bias",
    "ffn_up": "layer.1.DenseReluDense.wi_1",
    "ffn_down": "layer.1.DenseReluDense.wo",
    "ffn_gate": "layer.1.DenseReluDense.wi_0",
    "ffn_norm": "layer.1.layer_norm",
}

LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}

GEMMA3_SD_MAP = LLAMA_SD_MAP.copy()
GEMMA3_SD_MAP.update({
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
    "post_attention_norm": "post_attention_layernorm",
})

# Qwen3.5 linear-attention (GatedDeltaNet) tensors. llama.cpp names them ssm_* and fuses
# q/k/v into attn_qkv; ComfyUI wants linear_attn.* / self_attn.*. Matching is done per key
# against this map rather than by substring replace, which would rewrite attn_q inside
# attn_qkv and ssm_a inside ssm_alpha.

QWEN35_ATTN_MAP = {
    "attn_qkv": "linear_attn.in_proj_qkv",
    "attn_gate": "linear_attn.in_proj_z",
    "ssm_alpha": "linear_attn.in_proj_a",
    "ssm_beta": "linear_attn.in_proj_b",
    "ssm_out": "linear_attn.out_proj",
    "ssm_conv1d": "linear_attn.conv1d",
    "ssm_dt": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm": "linear_attn.norm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
}

CLIP_VISION_SD_MAP = {
    "mm.": "visual.merger.mlp.",
    "v.post_ln.": "visual.merger.ln_q.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}

QWEN3_VISION_SD_MAP = {
    "mm.0.": "visual.merger.linear_fc1.",
    "mm.2.": "visual.merger.linear_fc2.",
    "v.post_ln.": "visual.merger.norm.",
    "v.position_embd.": "visual.pos_embed.",
    "v.patch_embd": "visual.patch_embed.proj",
    "v.blk.": "visual.blocks.",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "attn_qkv.": "attn.qkv.",
    "attn_out.": "attn.proj.",
    "ln1.": "norm1.",
    "ln2.": "norm2.",
}


def sd_map_replace(raw_sd, key_map):
    sd = {}
    for k,v in raw_sd.items():
        for s,d in key_map.items():
            k = k.replace(s,d)
        sd[k] = v
    return sd

def llama_permute(raw_sd, n_head, n_head_kv):
    # Reverse version of LlamaModel.permute in llama.cpp convert script
    sd = {}
    permute = lambda x,h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)
    for k,v in raw_sd.items():
        if k.endswith(("q_proj.weight", "q_proj.bias")):
            v.data = permute(v.data, n_head)
        if k.endswith(("k_proj.weight", "k_proj.bias")):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd

def gemma3_norm_corrections(sd):
    # Reverse change from Gemma3Model modify_tensors in llama.cpp convert script
    norm_patterns = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "model.norm.weight"
    ]
    corrected = 0
    for key in list(sd.keys()):
        if any(p in key for p in norm_patterns):
            if is_quantized(sd[key]):
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float32) - 1.0
            else:
                sd[key] = sd[key].float() - 1.0
            corrected += 1
    #logging.info(f"Gemma3: Applied -1 norm correction to {corrected} tensors")
    return sd

def strip_quant_suffix(name):
    pattern = r"[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$"
    match = re.search(pattern, name, re.IGNORECASE)
    if match:
        name = name[:match.start()]
    return name

def gguf_mmproj_loader(path):
    # Reverse version of Qwen2VLVisionModel.modify_tensors
    logging.info("Attenpting to find mmproj file for text encoder...")

    # get name to match w/o quant suffix
    tenc_fname = os.path.basename(path)
    tenc = os.path.splitext(tenc_fname)[0].lower()
    tenc = strip_quant_suffix(tenc)

    # try and find matching mmproj
    target = []
    root = os.path.dirname(path)
    for fname in os.listdir(root):
        name, ext = os.path.splitext(fname)
        if ext.lower() != ".gguf":
            continue
        if "mmproj" not in name.lower():
            continue
        if tenc in name.lower():
            target.append(fname)

    if len(target) == 0:
        logging.error(f"Error: Can't find mmproj file for '{tenc_fname}' (matching:'{tenc}')! Qwen-Image-Edit will be broken!")
        return {}
    if len(target) > 1:
        logging.error(f"Ambiguous mmproj for text encoder '{tenc_fname}', will use first match.")

    logging.info(f"Using mmproj '{target[0]}' for text encoder '{tenc_fname}'.")
    target = os.path.join(root, target[0])
    vsd, extra = gguf_sd_loader(target, is_text_model=True)

    # concat 4D to 5D
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)

    # run main replacement
    key_map = CLIP_VISION_SD_MAP
    if extra["metadata"].get("clip.projector_type") == "qwen3vl_merger":
        key_map = QWEN3_VISION_SD_MAP.copy()
        deepstack_layers = sorted({int(k.split(".")[2]) for k in vsd if k.startswith("v.deepstack.")})
        for i, layer in enumerate(deepstack_layers):
            for src, dst in (("fc1", "linear_fc1"), ("fc2", "linear_fc2"), ("norm", "norm")):
                key_map[f"v.deepstack.{layer}.{src}."] = f"visual.deepstack_merger_list.{i}.{dst}."
    vsd = sd_map_replace(vsd, key_map)

    # handle split Q/K/V
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        # filter out attentions + group
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(
                    v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16)
                )

        # recombine
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([
                v[f"q.{suffix}"],
                v[f"k.{suffix}"],
                v[f"v.{suffix}"],
            ], dim=0)
        del attns

    return vsd

def gguf_tokenizer_loader(path, temb_shape):
    # convert gguf tokenizer to spiece
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please make sure sentencepiece and protobuf are installed.\npip install sentencepiece protobuf")
    spm = model.ModelProto()

    reader = gguf.GGUFReader(path)

    if get_field(reader, "tokenizer.ggml.model", str) == "t5":
        if temb_shape == (256384, 4096): # probably UMT5
            spm.trainer_spec.model_type == 1 # Unigram (do we have a T5 w/ BPE?)
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    spm.normalizer_spec.add_dummy_prefix = get_field(reader, "tokenizer.ggml.add_space_prefix", bool)
    spm.normalizer_spec.remove_extra_whitespaces = get_field(reader, "tokenizer.ggml.remove_extra_whitespaces", bool)

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    for idx, (token, score, toktype) in enumerate(zip(tokens, scores, toktypes)):
        # # These aren't present in the original?
        # if toktype == 5 and idx >= temb_shape[0]%1000):
        #     continue

        piece = spm.SentencePiece()
        piece.piece = token
        piece.score = score
        piece.type = toktype
        spm.pieces.append(piece)

    # unsure if any of these are correct
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.vocab_size = len(tokens) # split off unused?
    spm.trainer_spec.max_sentence_length = 4096
    spm.trainer_spec.eos_id = get_field(reader, "tokenizer.ggml.eos_token_id", int)
    spm.trainer_spec.pad_id = get_field(reader, "tokenizer.ggml.padding_token_id", int)

    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_tekken_tokenizer_loader(path, temb_shape):
    # convert ggml (hf) tokenizer metadata to tekken/comfy data
    logging.info("Attempting to recreate tekken tokenizer from GGUF file metadata...")
    import json
    import base64
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    reader = gguf.GGUFReader(path)

    model_str = get_field(reader, "tokenizer.ggml.model", str)
    if model_str == "gpt2":
        if temb_shape == (131072, 5120): # probably Mistral
            data = {
                "config": {"num_vocab_tokens": 150000, "default_vocab_size": 131072},
                "vocab": [],
                "special_tokens": [],
            }
        else:
            raise NotImplementedError("Unknown model, can't set tokenizer!")
    else:
        raise NotImplementedError("Unknown model, can't set tokenizer!")

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    toktypes = get_list_field(reader, "tokenizer.ggml.token_type", int)

    decoder = {v: k for k, v in bytes_to_unicode().items()}
    for idx, (token, toktype) in enumerate(zip(tokens, toktypes)):
        if toktype == 3:
            data["special_tokens"].append(
                {'rank': idx, 'token_str': token, 'is_control': True}
            )
        else:
            tok = bytes([decoder[char] for char in token])
            data["vocab"].append({
                "rank": len(data["vocab"]),
                "token_bytes": base64.b64encode(tok).decode("ascii"),
                "token_str": tok.decode("utf-8", errors="replace") # ?
            })

    logging.info(f"Created tekken tokenizer with vocab size of {len(data['vocab'])} (+{len(data['special_tokens'])})")
    del reader
    return torch.ByteTensor(list(json.dumps(data).encode('utf-8')))

def gguf_gemma3_tokenizer_loader(path):
    #TODO: merge into gguf_tokenizer_loader
    logging.info("Attempting to recreate sentencepiece tokenizer from GGUF file metadata...")
    try:
        from sentencepiece import sentencepiece_model_pb2 as model
    except ImportError:
        raise ImportError("Please install sentencepiece and protobuf.\npip install sentencepiece protobuf")
    spm = model.ModelProto()
    reader = gguf.GGUFReader(path)

    spm.normalizer_spec.name = "identity"
    spm.normalizer_spec.add_dummy_prefix = False
    spm.trainer_spec.model_type = 2
    spm.trainer_spec.input_format = "tsv"
    spm.trainer_spec.byte_fallback = True
    spm.trainer_spec.max_sentence_length = 4192
    spm.trainer_spec.bos_piece = "<bos>"

    tokens = get_list_field(reader, "tokenizer.ggml.tokens", str)
    scores = get_list_field(reader, "tokenizer.ggml.scores", float)
    toktype = get_list_field(reader, "tokenizer.ggml.token_type", int)
    
    if not tokens or not scores or not toktype:
        raise ValueError("Missing tokenizer metadata")
    
    for idx in range(len(tokens)):
        piece = spm.SentencePiece()
        piece.piece = tokens[idx]
        if idx == 3:  # UNK position
            piece.type = 2  # UNK Token
            piece.score = 0.0 # UNK Score
        else:
            piece.type = toktype[idx]
            piece.score = scores[idx]
        spm.pieces.append(piece)
    
    spm.trainer_spec.vocab_size = len(spm.pieces)
    logging.info(f"Created tokenizer with vocab size of {len(spm.pieces)}")
    
    del reader
    return torch.ByteTensor(list(spm.SerializeToString()))

def gguf_clip_loader(path):
    sd, extra = gguf_sd_loader(path, is_text_model=True)
    arch = extra.get("arch_str", None)
    if arch in {"t5", "t5encoder"}:
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape == (256384, 4096):
            # non-standard Comfy-Org tokenizer
            sd["spiece_model"] = gguf_tokenizer_loader(path, sd[temb_key].shape)
            # TODO: dequantizing token embed here is janky but otherwise we OOM due to tensor being massive.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        sd = sd_map_replace(sd, T5_SD_MAP)
    elif arch in {"llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3"}:
        # TODO: pass model_options["vocab_size"] to loader somehow
        temb_key = "token_embd.weight"
        if temb_key in sd and sd[temb_key].shape[0] >= (64 * 1024):
            if arch == "llama" and sd[temb_key].shape == (131072, 5120):
                # non-standard Comfy-Org tokenizer
                sd["tekken_model"] = gguf_tekken_tokenizer_loader(path, sd[temb_key].shape)
            elif arch == "gemma3":
                sd["spiece_model"] = gguf_gemma3_tokenizer_loader(path)
            # See note above for T5.
            logging.warning(f"Dequantizing {temb_key} to prevent runtime OOM.")
            sd[temb_key] = dequantize_tensor(sd[temb_key], dtype=torch.float16)
        if arch == "gemma3":
            sd = sd_map_replace(sd, GEMMA3_SD_MAP)
            sd = gemma3_norm_corrections(sd)
        else:
            sd = sd_map_replace(sd, LLAMA_SD_MAP)
        if arch == "llama":
            sd = llama_permute(sd, 32, 8) # L3 / Mistral
        if arch in {"qwen2vl", "qwen3vl"}:
            vsd = gguf_mmproj_loader(path)
            if arch == "qwen3vl":
                if not vsd:
                    raise ValueError(f"Qwen3-VL requires a matching mmproj GGUF alongside the text encoder: {path}")
                vsd = {f"model.{k}": v for k, v in vsd.items()}
            sd.update(vsd)
    elif arch == "qwen35":
        # Dequantize the two vocab-sized tensors up front. Left as GGMLTensor they are
        # re-dequantized on every forward, which both churns GBs per token and starves VRAM.
        for key in ("token_embd.weight", "output.weight"):
            if key in sd and sd[key].shape[0] >= (64 * 1024):
                logging.warning(f"Dequantizing {key} to prevent runtime OOM.")
                sd[key] = dequantize_tensor(sd[key], dtype=torch.float16)
        meta = extra.get("metadata", {})
        sd = qwen35_sd_remap(
            sd,
            num_k_heads=int(meta.get("qwen35.ssm.group_count", 16)),
            num_v_heads=int(meta.get("qwen35.ssm.time_step_rank", 32)),
            # convert_hf_to_gguf writes state_size from linear_key_head_dim.
            head_k_dim=int(meta.get("qwen35.ssm.state_size", 128)),
            head_v_dim=int(meta.get("qwen35.ssm.state_size", 128)),
        )
        vsd = gguf_mmproj_loader(path)
        if vsd:
            # sd.py strips the "model.visual." prefix before loading the vision tower.
            sd.update({f"model.{k}": v for k, v in vsd.items()})
    else:
        pass
    return sd

def qwen35_unreorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    """Undo llama.cpp's _reorder_v_heads (convert_hf_to_gguf.py, conversion/qwen.py).

    convert_hf_to_gguf regroups the linear-attention value heads from HF's per-key-head
    order into llama.cpp's tiled order whenever linear_num_key_heads != linear_num_value_heads.
    The operation swaps two axes, so applying it again restores the HF layout ComfyUI expects.
    """
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


def qwen35_sd_remap(sd, num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128):
    """llama.cpp qwen35 tensor names -> ComfyUI qwen35_* module names.

    Done per-key with explicit suffix matching: a flat substring replace would
    corrupt attn_q -> attn_qkv and ssm_a -> ssm_alpha.
    """
    out = {}
    for key, val in sd.items():
        if key == "token_embd.weight":
            new_key = "model.language_model.embed_tokens.weight"
        elif key == "output_norm.weight":
            new_key = "model.language_model.norm.weight"
        elif key == "output.weight":
            new_key = "lm_head.weight"
        elif not key.startswith("blk."):
            raise ValueError(f"Unexpected tensor name in qwen35 GGUF: {key}")
        else:
            layer, rest = key[4:].split(".", 1)
            prefix = f"model.language_model.layers.{layer}."
            if rest.startswith("attn_norm."):
                new_key = prefix + "input_layernorm." + rest[len("attn_norm."):]
            elif rest.startswith("post_attention_norm."):
                new_key = prefix + "post_attention_layernorm." + rest[len("post_attention_norm."):]
            elif rest.startswith("ffn_up."):
                new_key = prefix + "mlp.up_proj." + rest[len("ffn_up."):]
            elif rest.startswith("ffn_down."):
                new_key = prefix + "mlp.down_proj." + rest[len("ffn_down."):]
            elif rest.startswith("ffn_gate."):
                new_key = prefix + "mlp.gate_proj." + rest[len("ffn_gate."):]
            elif rest == "ssm_dt.bias":
                # llama.cpp names the dt bias as a bias; ComfyUI keeps it bare.
                new_key = prefix + "linear_attn.dt_bias"
            else:
                matched = False
                for src, dst in QWEN35_ATTN_MAP.items():
                    if rest == src or rest.startswith(src + "."):
                        new_key = prefix + dst + rest[len(src):]
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"Unexpected qwen35 layer tensor: {key}")
        out[new_key] = val

    # Undo llama.cpp's value-head regrouping for the GatedDeltaNet (linear attention) tensors.
    if num_k_heads and num_v_heads and num_k_heads != num_v_heads:
        num_v_per_k = num_v_heads // num_k_heads
        qk_dim = head_k_dim * num_k_heads

        def reorder(key, dim, head_dim, row_offset=0):
            tensor = out[key]
            if is_quantized(tensor):
                tensor = dequantize_tensor(tensor, dtype=torch.float16)
            else:
                tensor = tensor.to(dtype=torch.float16)
            if row_offset:
                head = tensor[:row_offset]
                tail = qwen35_unreorder_v_heads(tensor[row_offset:], dim, num_k_heads, num_v_per_k, head_dim)
                tensor = torch.cat([head, tail], dim=0) if dim == 0 else torch.cat([head, tail], dim=1)
            else:
                tensor = qwen35_unreorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim)
            out[key] = tensor

        for key in [k for k in out if k.endswith("linear_attn.in_proj_qkv.weight")]:
            reorder(key, 0, head_v_dim, row_offset=2 * qk_dim)
        for key in [k for k in out if k.endswith("linear_attn.in_proj_z.weight")]:
            reorder(key, 0, head_v_dim)
        for suffix in ("linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.A_log", "linear_attn.dt_bias"):
            for key in [k for k in out if k.endswith(suffix)]:
                reorder(key, 0, 1)
        for key in [k for k in out if k.endswith("linear_attn.out_proj.weight")]:
            reorder(key, 1, head_v_dim)

        # GatedDeltaNet conv1d: llama.cpp stores (channels, kernel), torch Conv1d wants
        # (out_channels, in_channels/groups, kernel). Only the value channels are regrouped.
        for key in [k for k in out if k.endswith("linear_attn.conv1d.weight")]:
            conv = out[key]
            if is_quantized(conv):
                conv = dequantize_tensor(conv, dtype=torch.float32)
            shape = getattr(conv, "tensor_shape", conv.shape)
            if len(shape) != 2:
                continue
            head, tail = conv[:2 * qk_dim], conv[2 * qk_dim:]
            tail = qwen35_unreorder_v_heads(tail, 0, num_k_heads, num_v_per_k, head_v_dim)
            out[key] = torch.cat([head, tail], dim=0).reshape(shape[0], 1, shape[1])

    # convert_hf_to_gguf writes A_log as -exp(A_log); ComfyUI applies the exp() itself.
    for key in [k for k in out if k.endswith("linear_attn.A_log")]:
        out[key] = torch.log(-out[key].to(dtype=torch.float32))

    # convert_hf_to_gguf folds the (1 + w) RMSNorm scaling into every norm weight except
    # linear_attn.norm; ComfyUI applies the 1.0 itself (rms_norm_add=True).
    for key in [k for k in out if k.endswith("weight") and "norm" in k and not k.endswith("linear_attn.norm.weight")]:
        out[key] = out[key].to(dtype=torch.float32) - 1.0
    return out