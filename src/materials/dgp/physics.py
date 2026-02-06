import numpy as np
import torch
import torch.nn as nn

try:
    import astra
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "astra-toolbox is required for CT operators. Install with `pip install astra-toolbox`."
    ) from exc


class _AstraForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, operator):
        ctx.operator = operator
        x_phys = (x.detach() + 1) / 2
        sino = operator._forward_tensor(x_phys, device=x.device, dtype=x.dtype)
        return sino

    @staticmethod
    def backward(ctx, grad_output):
        operator = ctx.operator
        grad_phys = operator._adjoint_tensor(
            grad_output.detach(), device=grad_output.device, dtype=grad_output.dtype
        )
        grad_input = grad_phys * 0.5
        return grad_input, None


class RadonTransform(nn.Module):
    """Parallel-beam CT operator backed by ASTRA (forward, adjoint, FBP)."""

    def __init__(self, image_size=128, num_angles=180, device="cuda"):
        super().__init__()
        self.image_size = int(image_size)
        self.num_angles = int(num_angles)
        self.device = device

        self.det_size = int(self.image_size * 1.5)
        self.angles = np.linspace(0, np.pi, self.num_angles, endpoint=False)
        self.proj_geom = astra.create_proj_geom(
            "parallel", 1.0, int(self.det_size), self.angles
        )
        self.vol_geom = astra.create_vol_geom(self.image_size, self.image_size)
        self.projector_id = astra.create_projector(
            "linear", self.proj_geom, self.vol_geom
        )

    def _forward_np(self, img):
        vol_id = astra.data2d.create("-vol", self.vol_geom, img)
        sino_id = astra.data2d.create("-sino", self.proj_geom)
        cfg = astra.astra_dict("FP")
        cfg["ProjectorId"] = self.projector_id
        cfg["VolumeDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        sino = astra.data2d.get(sino_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([vol_id, sino_id])
        return sino

    def _adjoint_np(self, sino):
        sino_id = astra.data2d.create("-sino", self.proj_geom, sino)
        vol_id = astra.data2d.create("-vol", self.vol_geom)
        cfg = astra.astra_dict("BP")
        cfg["ProjectorId"] = self.projector_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = vol_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        recon = astra.data2d.get(vol_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([vol_id, sino_id])
        return recon

    def _fbp_np(self, sino):
        sino_id = astra.data2d.create("-sino", self.proj_geom, sino)
        vol_id = astra.data2d.create("-vol", self.vol_geom)
        cfg = astra.astra_dict("FBP")
        cfg["ProjectorId"] = self.projector_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = vol_id
        cfg["FilterType"] = "ram-lak"
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        recon = astra.data2d.get(vol_id)
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([vol_id, sino_id])
        return recon

    def _forward_tensor(self, x_phys, device, dtype):
        b, c, h, w = x_phys.shape
        if h != self.image_size or w != self.image_size:
            raise ValueError(f"Input mismatch: {h}x{w} vs {self.image_size}")
        sinos = []
        for bi in range(b):
            chan_sinos = []
            for ci in range(c):
                img = x_phys[bi, ci].cpu().numpy().astype(np.float32)
                sino = self._forward_np(img)
                chan_sinos.append(torch.from_numpy(sino))
            sinos.append(torch.stack(chan_sinos, dim=0))
        return torch.stack(sinos, dim=0).to(device=device, dtype=dtype)

    def _adjoint_tensor(self, sinogram, device, dtype):
        b, c, angles, det = sinogram.shape
        if angles != self.num_angles or det != self.det_size:
            raise ValueError(
                f"Sinogram mismatch: {angles}x{det} vs {self.num_angles}x{self.det_size}"
            )
        recons = []
        for bi in range(b):
            chan_recons = []
            for ci in range(c):
                sino = sinogram[bi, ci].cpu().numpy().astype(np.float32)
                recon = self._adjoint_np(sino)
                chan_recons.append(torch.from_numpy(recon))
            recons.append(torch.stack(chan_recons, dim=0))
        return torch.stack(recons, dim=0).to(device=device, dtype=dtype)

    def _fbp_tensor(self, sinogram, device, dtype):
        b, c, angles, det = sinogram.shape
        if angles != self.num_angles or det != self.det_size:
            raise ValueError(
                f"Sinogram mismatch: {angles}x{det} vs {self.num_angles}x{self.det_size}"
            )
        recons = []
        for bi in range(b):
            chan_recons = []
            for ci in range(c):
                sino = sinogram[bi, ci].cpu().numpy().astype(np.float32)
                recon = self._fbp_np(sino)
                chan_recons.append(torch.from_numpy(recon))
            recons.append(torch.stack(chan_recons, dim=0))
        return torch.stack(recons, dim=0).to(device=device, dtype=dtype)

    def forward(self, x):
        """x: (B,C,H,W) in [-1, 1] -> sinogram in physics domain."""
        return _AstraForwardFunction.apply(x, self)

    def adjoint(self, sinogram):
        """Adjoint operator (backprojection), returns physics-domain image."""
        return self._adjoint_tensor(
            sinogram, device=sinogram.device, dtype=sinogram.dtype
        )

    def fbp(self, sinogram):
        """FBP reconstruction: sinogram (physics) -> image in [-1, 1]."""
        recon_phys = self._fbp_tensor(
            sinogram, device=sinogram.device, dtype=sinogram.dtype
        )
        recon_phys = recon_phys.clamp(0, 1)
        return (recon_phys * 2) - 1
