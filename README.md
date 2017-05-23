# cosmology
basic cosmology functions

The CMB class in cmb.py implements additional useful CMB functions through pycamb. The default LMAX=3500 and aboost=4 generates a very large (about 2gb) file for the transfer function data.

To set your own aboost and LMAX, do

```python
from cmb import CMB
planck15 = CMB(camb_init=False)
planck15.init_camb(aboost=4, LMAX=1000)
