import torch
import tempfile
from pathlib import Path
from PIL import Image


def to_uint8(images):
    """
    Convert [0,1] float tensor to uint8 [0,255]
    """
    images = (images * 255).clamp(0, 255).to(torch.uint8)
    return images


def save_batch_to_dir(batch, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(batch.size(0)):
        img = batch[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(out_dir / f"{i:05d}.png")


# FID + KID
def compute_metrics(real, generated, use_cuda=True):
    """
    Compute FID and KID using torch-fidelity.

    Args:
        real: [N,3,32,32] float in [0,1]
        generated: same

    Returns:
        fid, kid_mean, kid_std
    """
    import torch_fidelity

    real = to_uint8(real)
    generated = to_uint8(generated)

    with tempfile.TemporaryDirectory() as tmpdir:
        real_dir = Path(tmpdir) / "real"
        gen_dir = Path(tmpdir) / "gen"

        save_batch_to_dir(real, real_dir)
        save_batch_to_dir(generated, gen_dir)

        metrics = torch_fidelity.calculate_metrics(
            input1=str(real_dir),
            input2=str(gen_dir),
            cuda=use_cuda,
            fid=True,
            kid=True,
            kid_subsets=50,
            kid_subset_size=100,
            isc=False,
            verbose=False,
        )

    return (
        float(metrics["frechet_inception_distance"]),
        float(metrics["kernel_inception_distance_mean"]),
        float(metrics["kernel_inception_distance_std"]),
    )