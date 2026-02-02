from pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import torch
import gc
import os

# 1. Dọn dẹp bộ nhớ
torch.cuda.empty_cache()
gc.collect()

print(f"CUDA available: {torch.cuda.is_available()}")

try:
    # Load model (Nên dùng bản difix_ref để tối ưu cho việc dùng ảnh tham chiếu)
    pipe = DifixPipeline.from_pretrained(
        "nvidia/difix_ref", 
        trust_remote_code=True, 
        torch_dtype=torch.float16
    )
    
    pipe.enable_model_cpu_offload() 
    print("Load model thành công!")

    # 2. KHAI BÁO ĐƯỜNG DẪN ẢNH (Input và Reference)
    input_path = "../assets/example_input.png"
    ref_path = "../assets/example_ref.png"  # <--- File bạn mới thêm vào

    # Kiểm tra file input có tồn tại không
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file input tại {input_path}")
    
    # Load ảnh Input
    input_image = load_image(input_path)
    
    # Load ảnh Reference (Quan trọng)
    if os.path.exists(ref_path):
        print(f"Đã tìm thấy ảnh tham chiếu: {ref_path}")
        ref_image = load_image(ref_path)
    else:
        # Fallback: Nếu lỡ không tìm thấy file ref thì dùng tạm input làm ref để code không sập
        print(f"Cảnh báo: Không tìm thấy {ref_path}. Đang dùng Input làm Reference tạm thời.")
        ref_image = input_image

    prompt = "remove degradation, high quality"

    print("Đang xử lý với ảnh tham chiếu...")
    
    # 3. CHẠY PIPELINE VỚI THAM SỐ REF_IMAGE
    output_image = pipe(
        prompt, 
        image=input_image, 
        ref_image=ref_image, # <--- BẮT BUỘC THÊM DÒNG NÀY
        num_inference_steps=1, 
        timesteps=[199], 
        guidance_scale=0.0
    ).images[0]
    
    output_image.save("../assets/example_output.png")
    print("CHẠY THÀNH CÔNG! Ảnh kết quả đã lưu tại example_output.png")

except Exception as e:
    print(f"\nCó lỗi xảy ra: {e}")