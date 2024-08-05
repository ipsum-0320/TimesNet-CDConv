# TimesNet-CDConv

This is the repo of TimesNet-CDConv.

Thanks to the following repo:
* Time-Series-Library([https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library))

## For Mac m1 users:

if you meet the error: `RuntimeError: fft: ATen not compiled with MKL support`。

please check [issue](https://github.com/pytorch/pytorch/issues/63592), you can replace your torch, run follows cmd:

```bash
pip uninstall torch

pip install --pre torch -f https://download.pytorch.org/whl/nightly/torch_nightly.html 
```