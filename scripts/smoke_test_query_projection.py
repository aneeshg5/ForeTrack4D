import torch

from foretrack.models.conditioning import project_to_vit_frame, sample_patch_features


def main():
    vit_input_size = (256, 192)  # (H, W)
    h, w = vit_input_size

    col = torch.linspace(0, 1, w).reshape(1, 1, 1, w).expand(1, 1, h, w)
    row = torch.linspace(0, 1, h).reshape(1, 1, h, 1).expand(1, 1, h, w)
    patch_feats = torch.cat([col, row], dim=1)  # (1, 2, H, W)

    orig_h, orig_w = 480, 640  # a real DexYCB frame size, deliberately not square

    def sample_at(u, v):
        uv = torch.tensor([[[u, v]]], dtype=torch.float32)
        uv_vit = project_to_vit_frame(uv, (orig_h, orig_w), vit_input_size)
        return sample_patch_features(patch_feats, uv_vit, vit_input_size)[0, 0]

    center = sample_at(orig_w / 2, orig_h / 2)
    print(f"center sample: {center.tolist()}")
    assert abs(center[0].item() - 0.5) < 0.05, f"center column sample off: {center[0].item()}"
    assert abs(center[1].item() - 0.5) < 0.05, f"center row sample off: {center[1].item()}"

    left = sample_at(orig_w * 0.3, orig_h / 2)
    right = sample_at(orig_w * 0.7, orig_h / 2)
    print(f"left={left.tolist()} right={right.tolist()}")
    assert left[0].item() < center[0].item() < right[0].item(), "column sampling isn't monotonic in u"

    top = sample_at(orig_w / 2, orig_h * 0.3)
    bottom = sample_at(orig_w / 2, orig_h * 0.7)
    print(f"top={top.tolist()} bottom={bottom.tolist()}")
    assert top[1].item() < center[1].item() < bottom[1].item(), "row sampling isn't monotonic in v"

    print("SMOKE_TEST_PASSED")


if __name__ == "__main__":
    main()
