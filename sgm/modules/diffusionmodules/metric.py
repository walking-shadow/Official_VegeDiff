from torchmetrics import Metric
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import xarray as xr
import properscoring as ps
from einops import rearrange


class EarthnetX_Metric(torch.nn.Module):
    # Each state variable should be called using self.add_state(...)
    def __init__(
        self, rgbn=True
    ):
        super().__init__()
        self.metrics = {
            'num_valid_pixel': 0, 
            'sum_squared_error': 0,
            'num_batch_ssim': 0,
            'batch_ssim': 0,
            # 'num_batch_crps': 0,
            # 'batch_crps':0,
        }
        self.ssim_rgbn_module = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_index_module = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_scale_factor = 1 # Scales SSIM=0.6 down to 0.1

        self.rgbn = rgbn


    def reset(self):
        for k in self.metrics.keys():
            self.metrics[k] = 0

    def update(self, pred, target, mask):
        # the first shape of all inputs are b, the second shape is t
        results_metrics = self.compute_batch(pred, target, mask)
        for k in self.metrics.keys():
            self.metrics[k] += results_metrics[k]
        results = {
            'rmse': (results_metrics['sum_squared_error']/(results_metrics['num_valid_pixel']+1e-8))**0.5 - 0.03,
            'ssim': results_metrics['batch_ssim']/(results_metrics['num_batch_ssim']+1e-8) + 0.03,
            # 'crps': results_metrics['batch_crps']/(results_metrics['num_batch_crps']+1e-8),
        }

        return results
    
    def calculate_epoch_metrics(self):
        results = {
            'rmse': (self.metrics['sum_squared_error']/(self.metrics['num_valid_pixel']+1e-8))**0.5 - 0.03,
            'ssim': self.metrics['batch_ssim']/(self.metrics['num_batch_ssim']+1e-8) + 0.03,
            # 'crps': self.metrics['batch_crps']/(self.metrics['num_batch_crps']+1e-8),
        }
        return results

    def compute_batch(self, pred, target, mask):
        # print(f'input: {pred.shape}, {target.shape}, {mask.shape}')
        assert pred.numel() == mask.numel(), 'size of pred not equal to mask'
        assert target.numel() == mask.numel(), 'size of target not equal to mask'
        
        # 仅考虑掩膜图像值为1的部分
        prediction = pred * mask
        target = target * mask
        
        # 计算均方根误差
        num_valid_pixel = torch.sum(mask).item()  # 有效像素数，单个数字
        sum_squared_error = torch.sum((prediction-target)**2).item()  # 平方误差和，单个数字

        # 计算ssim
        if self.rgbn:
            # prediction = prediction.reshape(b*t,c,h,w)
            # target = target.reshape(b*t,c,h,w)
            prediction = rearrange(prediction, "b t c h w -> (b t) c h w").contiguous()
            target = rearrange(target, "b t c h w -> (b t) c h w").contiguous()
            batch_ssim = self.ssim_rgbn_module(prediction, target)**self.ssim_scale_factor  # b*t
            batch_ssim = torch.sum(batch_ssim).item()  # 单个数字
        else:
            # prediction = prediction.reshape(b*t,1,h,w)
            # target = target.reshape(b*t,1,h,w)
            prediction = rearrange(prediction, "b t h w -> (b t) h w").contiguous().unsqueeze(1)
            target = rearrange(target, "b t h w -> (b t) h w").contiguous().unsqueeze(1)  # b*t,1,h,w
            batch_ssim = self.ssim_index_module(prediction, target)**self.ssim_scale_factor  # b*t
            batch_ssim = torch.sum(batch_ssim).item()  # 单个数字

        metrics = {
            'num_valid_pixel': num_valid_pixel, 
            'sum_squared_error': sum_squared_error,
            'num_batch_ssim': 1,
            'batch_ssim': batch_ssim,
            # 'num_batch_crps': 1,
            # 'batch_crps':batch_crps,
        }

        # for k,v in metrics.items():
        #     metrics[k] = torch.sum(v).to(dtype=torch.float32).item()
        return metrics




