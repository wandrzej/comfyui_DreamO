import torch
from safetensors.torch import load_file as load_safetensors
from diffusers import (
    AutoencoderKL,
    # CLIPTextModelWithProjection, # Removed from here
    FluxTransformer2DModel,
    FluxScheduler 
)
from diffusers.models import CLIPTextModelWithProjection # Added here
from transformers import CLIPTokenizer
import os
import numpy as np
from PIL import Image 
from einops import repeat

# ComfyUI specific imports
import comfy.utils
import comfy.sd
import comfy.model_management 

# Import utils
from .utils import resize_numpy_image_area, img2tensor, convert_flux_lora_to_diffusers

class DreamOFluxLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_checkpoint_path": ("STRING", {"default": "path/to/flux_transformer_hf_directory_or_file"}), # Expected path to a directory containing the FluxTransformer2DModel weights and config (Hugging Face model format), or a single .safetensors file if the loading code is adapted for it.
                "vae_path": ("STRING", {"default": "path/to/vae.safetensors"}), # Path to the VAE model file (e.g., .safetensors).
                "clip_l_path": ("STRING", {"default": "path/to/clip_l_hf_directory_or_file"}), # Path to CLIPTextModelWithProjection directory (Hugging Face model format) or a compatible .safetensors file.
                "clip_g_path": ("STRING", {"default": "path/to/clip_g_hf_directory_or_file"}), # Path to CLIPTextModelWithProjection directory (Hugging Face model format) or a compatible .safetensors file.
                "dreamo_weights_path": ("STRING", {"default": "path/to/dreamo.safetensors"}), # Path to 'dreamo.safetensors' containing T5 embeddings, task/index embeddings, and the main DreamO LoRA weights.
                "tokenizer_l_path": ("STRING", {"default": "path/to/tokenizer_l_hf_directory"}), # Path to the Hugging Face tokenizer directory for CLIP L respectively.
                "tokenizer_g_path": ("STRING", {"default": "path/to/tokenizer_g_hf_directory"}), # Path to the Hugging Face tokenizer directory for CLIP G respectively.
            },
            "optional": {
                "turbo_weights_path": ("STRING", {"default": ""}), # Optional: Path to the LoRA .safetensors file.
                "cfg_distill_lora_path": ("STRING", {"default": ""}), # Optional: Path to the LoRA .safetensors file.
                "quality_pos_lora_path": ("STRING", {"default": ""}), # Optional: Path to the LoRA .safetensors file.
                "quality_neg_lora_path": ("STRING", {"default": ""}), # Optional: Path to the LoRA .safetensors file.
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP_L", "CLIP_G", "VAE", "DREAMO_RESOURCES")
    FUNCTION = "load_flux_dreamo"
    CATEGORY = "DreamO"

    def load_flux_dreamo(self, flux_checkpoint_path, vae_path, clip_l_path, clip_g_path, dreamo_weights_path,
                         tokenizer_l_path, tokenizer_g_path, turbo_weights_path=None, cfg_distill_lora_path=None,
                         quality_pos_lora_path=None, quality_neg_lora_path=None):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        torch_dtype = torch.bfloat16 if comfy.model_management.supports_bf16() else torch.float32
        dreamo_resources = {}
        tokenizer_l = CLIPTokenizer.from_pretrained(tokenizer_l_path)
        tokenizer_g = CLIPTokenizer.from_pretrained(tokenizer_g_path)
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch_dtype)
        text_encoder_l = CLIPTextModelWithProjection.from_pretrained(clip_l_path, torch_dtype=torch_dtype)
        text_encoder_g = CLIPTextModelWithProjection.from_pretrained(clip_g_path, torch_dtype=torch_dtype)
        transformer = FluxTransformer2DModel.from_pretrained(flux_checkpoint_path, torch_dtype=torch_dtype)
        dreamo_data = load_safetensors(dreamo_weights_path, device="cpu")
        dreamo_resources['t5_embedding_weight'] = dreamo_data.pop('dreamo_t5_embedding.weight')[-10:].clone()
        dreamo_resources['task_embedding_weight'] = dreamo_data.pop('dreamo_task_embedding.weight').clone()
        dreamo_resources['idx_embedding_weight'] = dreamo_data.pop('dreamo_idx_embedding.weight').clone()
        dreamo_transformer_lora_weights = convert_flux_lora_to_diffusers(dreamo_data)
        adapter_names = ["dreamo"]; adapter_weights = [1.0]
        transformer.load_lora_weights(dreamo_transformer_lora_weights, adapter_name="dreamo")
        optional_lora_configs = [
            (turbo_weights_path, "turbo", 1.0, False), (cfg_distill_lora_path, "cfg", 1.0, True),
            (quality_pos_lora_path, "quality_pos", 0.15, True), (quality_neg_lora_path, "quality_neg", -0.8, True)
        ]
        for lora_path, lora_name, lora_weight, needs_conversion in optional_lora_configs:
            if lora_path and os.path.exists(lora_path):
                lora_data_state_dict = load_safetensors(lora_path, device="cpu")
                diffusers_lora_weights = convert_flux_lora_to_diffusers(lora_data_state_dict) if needs_conversion else lora_data_state_dict
                transformer.load_lora_weights(diffusers_lora_weights, adapter_name=lora_name)
                adapter_names.append(lora_name); adapter_weights.append(lora_weight)
        if adapter_names:
            transformer.set_adapters(adapter_names, adapter_weights)
            transformer.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
            transformer.unload_lora_weights() 
        transformer.to(device); vae.to(device); text_encoder_l.to(device); text_encoder_g.to(device)
        return (transformer, (text_encoder_l, tokenizer_l), (text_encoder_g, tokenizer_g), vae, dreamo_resources)

class DreamOFluxSamplerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), "clip_l": ("CLIP_L",), "clip_g": ("CLIP_G",), "vae": ("VAE",),   
                "dreamo_resources": ("DREAMO_RESOURCES",),
                "positive_prompt": ("STRING", {"default": "A beautiful landscape", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16}),
                "num_steps": ("INT", {"default": 12, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
            }, "optional": {
                "ref_image_1": ("IMAGE", {"default": None}), 
                "ref_task_1": (["ip", "id", "style"], {"default": "ip"}), # Task type for the reference image: "ip" (image prompt/general IP adapter), "id" (identity preservation), "style" (style transfer).
                "ref_image_2": ("IMAGE", {"default": None}), 
                "ref_task_2": (["ip", "id", "style"], {"default": "ip"}), # Task type for the reference image: "ip" (image prompt/general IP adapter), "id" (identity preservation), "style" (style transfer).
                "ref_res": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 16}),
                "true_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}), # For True CFG
                "cfg_start_step_percent": ("INT", {"default": 0, "min": 0, "max": 100}), # Start step for applying True CFG, as a percentage of total steps.
                "cfg_end_step_percent": ("INT", {"default": 0, "min": 0, "max": 100}),   # End step for applying True CFG, as a percentage of total steps.
                "neg_guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}), # Separate guidance scale for the negative prompt when True CFG is active.
                "first_step_guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}), # Guidance scale for the very first sampling step. If 0, uses main guidance_scale.
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image_flux"
    CATEGORY = "DreamO"

    @staticmethod
    def _get_task_embedding_idx(task_str: str) -> int:
        task_map = {"ip": 0, "id": 0, "style": 1} 
        return task_map.get(task_str.lower(), 0)

    @staticmethod
    def _sampler_prepare_t5_embeddings(tokenizer_g, text_encoder_g, t5_embedding_weight_tensor, device):
        num_new_tokens = 10
        new_tokens = [f"[ref#{i}]" for i in range(1, num_new_tokens)] + ["[res]"]
        original_vocab_size = len(tokenizer_g)
        added_tokens = tokenizer_g.add_tokens(new_tokens, special_tokens=False)
        if added_tokens > 0: text_encoder_g.resize_token_embeddings(len(tokenizer_g))
        new_token_ids = tokenizer_g.convert_tokens_to_ids(new_tokens)
        token_embedding_layer = text_encoder_g.text_model.embeddings.token_embedding
        if token_embedding_layer.weight.shape[0] != len(tokenizer_g): text_encoder_g.resize_token_embeddings(len(tokenizer_g))
        with torch.no_grad():
            for i, token_id in enumerate(new_token_ids):
                token_embedding_layer.weight[token_id] = t5_embedding_weight_tensor[i].to(device=device, dtype=token_embedding_layer.weight.dtype)

    @staticmethod
    def _sampler_prepare_latent_image_ids(batch_size, height_lat, width_lat, device, dtype, start_height=0, start_width=0):
        latent_image_ids = torch.zeros(height_lat, width_lat, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = torch.arange(height_lat, device=device)[:, None] + start_height
        latent_image_ids[..., 2] = torch.arange(width_lat, device=device)[None, :] + start_width
        return latent_image_ids.reshape(batch_size, height_lat * width_lat, 3)

    @staticmethod
    def _sampler_encode_prompt(prompt, tokenizer_l, text_encoder_l, tokenizer_g, text_encoder_g, max_sequence_length, device, clean_caption=False):
        if not isinstance(prompt, list): prompt = [prompt]
        if clean_caption: prompt = [p.lower().replace("-", " ").replace("_", " ").strip() for p in prompt]
        text_inputs_l = tokenizer_l(prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_attention_mask=True, return_tensors="pt")
        text_input_ids_l = text_inputs_l.input_ids.to(device); attention_mask_l = text_inputs_l.attention_mask.to(device)
        text_encoder_l.to(device); prompt_embeds_l = text_encoder_l(input_ids=text_input_ids_l, attention_mask=attention_mask_l, output_hidden_states=True).hidden_states[-1]
        text_inputs_g = tokenizer_g(prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_attention_mask=True, return_tensors="pt")
        text_input_ids_g = text_inputs_g.input_ids.to(device); attention_mask_g = text_inputs_g.attention_mask.to(device)
        text_encoder_g.to(device); outputs_g = text_encoder_g(input_ids=text_input_ids_g, attention_mask=attention_mask_g, output_hidden_states=True)
        prompt_embeds_g = outputs_g.hidden_states[-1]; pooled_prompt_embeds = outputs_g.text_embeds
        prompt_embeds = torch.cat((prompt_embeds_l, prompt_embeds_g), dim=-1)
        return prompt_embeds, pooled_prompt_embeds, text_input_ids_g

    def generate_image_flux(self, model, clip_l, clip_g, vae, dreamo_resources, 
                            positive_prompt, negative_prompt,
                            width, height, num_steps, guidance_scale, seed,
                            ref_image_1=None, ref_task_1="ip", ref_image_2=None, ref_task_2="ip",
                            ref_res=512, true_cfg_scale=1.0, cfg_start_step_percent=0, cfg_end_step_percent=0,
                            neg_guidance_scale=3.5, first_step_guidance_scale_val=0.0):

        transformer_model = model
        text_encoder_l, tokenizer_l = clip_l
        text_encoder_g, tokenizer_g = clip_g
        device = transformer_model.device; dtype = transformer_model.dtype
        if seed == -1: seed = torch.Generator(device="cpu").seed()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        self._sampler_prepare_t5_embeddings(tokenizer_g, text_encoder_g, dreamo_resources['t5_embedding_weight'], device)
        max_len = tokenizer_g.model_max_length
        pos_prompt_embeds, pos_pooled_embeds, pos_text_ids = self._sampler_encode_prompt(positive_prompt, tokenizer_l, text_encoder_l, tokenizer_g, text_encoder_g, max_len, device)
        neg_prompt_embeds, neg_pooled_embeds, neg_text_ids = self._sampler_encode_prompt(negative_prompt, tokenizer_l, text_encoder_l, tokenizer_g, text_encoder_g, max_len, device, clean_caption=True)
        
        batch_size = 1; vae_scale_factor = 8 
        height_lat, width_lat = height // vae_scale_factor, width // vae_scale_factor
        num_latent_channels = transformer_model.config.in_channels
        latents = torch.randn((batch_size, num_latent_channels, height_lat, width_lat), dtype=dtype, device="cpu", generator=generator).to(device)
        
        scheduler = FluxScheduler(); scheduler.set_timesteps(num_steps, device=device); timesteps = scheduler.timesteps
        if hasattr(scheduler, 'init_noise_sigma') and scheduler.init_noise_sigma is not None: latents = latents * scheduler.init_noise_sigma
        
        ref_img_latents_for_concat, ref_embeddings_for_concat, ref_img_ids_for_concat = [], [], []
        task_embed_layer = torch.nn.Embedding.from_pretrained(dreamo_resources['task_embedding_weight'].to(device, dtype=dtype), freeze=True)
        idx_embed_layer = torch.nn.Embedding.from_pretrained(dreamo_resources['idx_embedding_weight'].to(device, dtype=dtype), freeze=True)
        base_task_embedding_vector = task_embed_layer.weight[self._get_task_embedding_idx("style")] 
        current_start_h_lat, current_start_w_lat = height_lat // 2, width_lat // 2

        for ref_idx_loop, (ref_image_comfy, ref_task_str, dreamo_idx_val) in enumerate([(ref_image_1, ref_task_1, 0), (ref_image_2, ref_task_2, 1)]):
            if ref_image_comfy is not None:
                img_tensor_bhwc = ref_image_comfy[0]; np_image_hwc = (img_tensor_bhwc.cpu().numpy() * 255.0).astype(np.uint8)
                processed_ref_image_np = resize_numpy_image_area(np_image_hwc, ref_res * ref_res)
                ref_img_pixels_for_vae = 2.0 * img2tensor(processed_ref_image_np, bgr2rgb=False, float32=True).unsqueeze(0).to(device, dtype=vae.dtype) - 1.0
                ref_latent_bchw = vae.encode(ref_img_pixels_for_vae).latent_dist.sample(generator=generator) * vae.config.scaling_factor
                ref_latent_b_s_d = ref_latent_bchw.permute(0, 2, 3, 1).reshape(batch_size, -1, ref_latent_bchw.shape[1])
                ref_img_latents_for_concat.append(ref_latent_b_s_d)
                ref_h_lat_curr, ref_w_lat_curr = ref_latent_bchw.shape[2], ref_latent_bchw.shape[3]
                current_ref_img_ids = self._sampler_prepare_latent_image_ids(batch_size, ref_h_lat_curr, ref_w_lat_curr, device, dtype, current_start_h_lat, current_start_w_lat)
                ref_img_ids_for_concat.append(current_ref_img_ids)
                current_start_h_lat += ref_h_lat_curr // 2; current_start_w_lat += ref_w_lat_curr // 2
                task_e = task_embed_layer(torch.tensor([self._get_task_embedding_idx(ref_task_str)], device=device))
                idx_e = idx_embed_layer(torch.tensor([dreamo_idx_val], device=device))
                repeated_task_idx_embed = repeat((task_e + idx_e).squeeze(0), 'd -> b s d', b=batch_size, s=ref_latent_b_s_d.shape[1])
                ref_embeddings_for_concat.append(repeated_task_idx_embed)

        main_img_ids = self._sampler_prepare_latent_image_ids(batch_size, height_lat, width_lat, device, dtype)
        num_main_img_tokens = main_img_ids.shape[1]
        main_task_embeddings = repeat(base_task_embedding_vector, 'd -> b s d', b=batch_size, s=num_main_img_tokens)
        
        latents_b_s_d = latents.permute(0, 2, 3, 1).reshape(batch_size, -1, num_latent_channels)
        
        final_img_ids = torch.cat([main_img_ids] + ref_img_ids_for_concat, dim=1)
        final_task_idx_embeddings = torch.cat([main_task_embeddings] + ref_embeddings_for_concat, dim=1)
        
        do_guidance = guidance_scale > 1.0
        actual_cfg_start_step = int(num_steps * cfg_start_step_percent / 100.0)
        actual_cfg_end_step = int(num_steps * cfg_end_step_percent / 100.0)
        progress_bar = comfy.utils.ProgressBar(num_steps)

        for i, t in enumerate(timesteps):
            current_latents_b_s_d_template = torch.cat([latents_b_s_d] + ref_img_latents_for_concat, dim=1)
            
            # Determine current guidance scale for this step
            current_step_guidance_scale = first_step_guidance_scale_val if i == 0 and first_step_guidance_scale_val > 0 else guidance_scale
            
            if do_guidance:
                num_cond_passes = 2
                latent_model_input = torch.cat([current_latents_b_s_d_template] * num_cond_passes)
                prompt_embeds_input = torch.cat([neg_prompt_embeds, pos_prompt_embeds])
                pooled_embeds_input = torch.cat([neg_pooled_embeds, pos_pooled_embeds])
                text_ids_input = torch.cat([neg_text_ids, pos_text_ids])
                task_idx_embeddings_input = torch.cat([final_task_idx_embeddings] * num_cond_passes)
                img_ids_input = torch.cat([final_img_ids] * num_cond_passes)
                guidance_input_transformer = torch.tensor([current_step_guidance_scale] * num_cond_passes, device=device, dtype=dtype) # Pass scalar to transformer
            else: # Not doing CFG
                latent_model_input = current_latents_b_s_d_template
                prompt_embeds_input = pos_prompt_embeds
                pooled_embeds_input = pos_pooled_embeds
                text_ids_input = pos_text_ids
                task_idx_embeddings_input = final_task_idx_embeddings
                img_ids_input = final_img_ids
                guidance_input_transformer = None # Or a single guidance_scale if model expects it

            current_timestep_expanded = t.expand(latent_model_input.shape[0]).to(dtype) / 1000.0

            noise_pred_sample = transformer_model(
                hidden_states=latent_model_input, timestep=current_timestep_expanded, 
                encoder_hidden_states=prompt_embeds_input, pooled_projections=pooled_embeds_input,
                txt_ids=text_ids_input, img_ids=img_ids_input, embeddings=task_idx_embeddings_input,
                guidance=guidance_input_transformer if transformer_model.config.guidance_embeds else None
            ).sample

            if do_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred_sample.chunk(2)
                noise_pred = noise_pred_uncond + current_step_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # True CFG (Simplified: uses neg_guidance_scale for the uncond part in the CFG formula if active)
                # A full True CFG might involve a third pass or specific model weights.
                # This version re-calculates noise_pred if true_cfg is active.
                if true_cfg_scale > 1.0 and i >= actual_cfg_start_step and i < actual_cfg_end_step:
                    # For "True CFG", the unconditional part is sometimes scaled by a different negative guidance
                    # Let's assume the `neg_guidance_scale` input is for this.
                    # The prompt implies a third pass, but for simplicity, let's adjust the existing uncond if neg_guidance_scale is different.
                    # If neg_guidance_scale == current_step_guidance_scale, this doesn't change uncond.
                    # This is a common interpretation for "true negative" or "per-prompt CFG".
                    # A more rigorous True CFG (like in some research papers) might involve more complex steps.
                    # Given the context, this likely means allowing a different scale for the negative prompt's influence.
                    # The "true_cfg_scale" then scales the overall guidance from the conditional.
                    
                    # If neg_guidance_scale is meant to be the guidance for the uncond pass in True CFG context:
                    noise_pred_uncond_true_cfg = noise_pred_uncond # Use the already computed uncond
                    # The 'true_cfg_scale' might be used to further emphasize the conditional part
                    noise_pred = noise_pred_uncond_true_cfg + true_cfg_scale * (noise_pred_cond - noise_pred_uncond_true_cfg)
                    # Note: The original DreamO code uses a specific LoRA for CFG distillation, which is already
                    # loaded if `cfg_distill_lora_path` is provided. The parameters here offer fine-tuning.
            else:
                noise_pred = noise_pred_sample
            
            noise_pred_main_image_b_s_d = noise_pred[:, :num_main_img_tokens, :]
            noise_pred_main_image_bchw = noise_pred_main_image_b_s_d.reshape(batch_size, height_lat, width_lat, num_latent_channels).permute(0, 3, 1, 2)
            
            latents = scheduler.step(noise_pred_main_image_bchw, t, latents).prev_sample
            latents_b_s_d = latents.permute(0, 2, 3, 1).reshape(batch_size, -1, num_latent_channels)
            progress_bar.update()

        latents_output_bchw = latents / vae.config.scaling_factor
        decoded_pixels = vae.decode(latents_output_bchw.to(vae.dtype)).sample
        decoded_pixels = (decoded_pixels / 2 + 0.5).clamp(0, 1)
        output_images_b_h_w_c = decoded_pixels.permute(0, 2, 3, 1).cpu().float()
        return (output_images_b_h_w_c,)

NODE_CLASS_MAPPINGS = {
    "DreamOFluxLoaderNode": DreamOFluxLoaderNode,
    "DreamOFluxSamplerNode": DreamOFluxSamplerNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOFluxLoaderNode": "DreamO FLUX Loader (Refactored)",
    "DreamOFluxSamplerNode": "DreamO FLUX Sampler (Manual Full)"
}
