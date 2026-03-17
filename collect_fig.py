import os
import shutil


def collect():
    prefix = "20241127-2352-azurecode-pick-9"
    for m in ["llama2_7b", "llama2_7b_gqa"]:
        model_name = m
        for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            prefill_upper_bound = p
            for t in [0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000]:
                mock_scale_millis = t
                for i in [1, 2, 3, 4]:
                    if model_name == "llama2_7b":
                        tag_path = f"mha/prefill-{prefill_upper_bound}/mock-{mock_scale_millis}/{i}"
                        tag_name = f"mha-prefill-{prefill_upper_bound}-mock-{mock_scale_millis}-{i}"
                    else:
                        tag_path = f"gqa/prefill-{prefill_upper_bound}/mock-{mock_scale_millis}/{i}"
                        tag_name = f"gqa-prefill-{prefill_upper_bound}-mock-{mock_scale_millis}-{i}"
                    shutil.copy(
                        f"./log/{prefix}/{tag_path}/fig.pdf",
                        f"./output/{prefix}/{tag_name}.pdf",
                    )


if __name__ == "__main__":
    collect()
