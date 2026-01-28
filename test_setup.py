from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import torch
import gc

# 1. Dọn dẹp bộ nhớ cũ
torch.cuda.empty_cache()
gc.collect()

print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # --- SỬA LẠI ĐOẠN NÀY ---
    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix", 
        trust_remote_code=True, 
        torch_dtype=torch.float16  # Vẫn giữ dòng này để ép model chạy nhẹ (tiết kiệm VRAM)
        # ĐÃ XÓA dòng variant="fp16" gây lỗi
    )
    
    # 2. Bật chế độ tiết kiệm VRAM tối đa cho GPU 8GB
    pipe.enable_model_cpu_offload() 
    
    print("Load model thành công, bắt đầu xử lý ảnh...")

    input_image = load_image("assets/example_input.png")
    prompt = "remove degradation"

    # 3. Chạy thử
    output_image = pipe(
        prompt, 
        image=input_image, 
        num_inference_steps=1, 
        timesteps=[199], 
        guidance_scale=0.0
    ).images[0]
    
    output_image.save("example_output.png")
    print("CHẠY THÀNH CÔNG! Ảnh kết quả đã lưu tại example_output.png")

except Exception as e:
    print(f"\nVẫn gặp lỗi: {e}")