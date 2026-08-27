import os
os.environ['KARAONE_AUTO_INSTALL']='0'
os.environ['KARAONE_HYDRA_KERNELS']='2'
os.environ['KARAONE_HYDRA_GROUPS']='4'
os.environ['KARAONE_ROCKET_KERNELS']='32'
import numpy as np
from karaone_final_kaggle import preprocess_epoch, qc_fraction, safe_stat_features, StaticModel, AeonModel, MiniRocketHydraFusion
rng=np.random.default_rng(42)
X=rng.normal(size=(24,62,1280)).astype('float32')
for mode in ['asr_notch_car','official']:
    os.environ['KARAONE_PREPROCESSING_MODE']=mode
    out=preprocess_epoch(X[0],1000.0)
    assert out.shape==(62,1280)
assert 0.0 <= qc_fraction(X[0],1000.0) <= 1.0
F=safe_stat_features(X)
assert F.shape[0]==24 and np.isfinite(F).all()
model=StaticModel('summary').fit(X, np.array([0,1]*12))
p=model.score(X)
assert p.shape==(24,) and np.isfinite(p).all()
for name in ['hydra','mr_hydra','minirocket','multirocket']:
    m=AeonModel(name).fit(X, np.array([0,1]*12))
    p=m.score(X)
    assert p.shape==(24,) and np.isfinite(p).all()
for kind in ['catch22','summary','catch22_summary']:
    m=StaticModel(kind).fit(X, np.array([0,1]*12))
    p=m.score(X)
    assert p.shape==(24,) and np.isfinite(p).all()
f=MiniRocketHydraFusion().fit(X, np.array([0,1]*12))
p=f.score(X)
assert p.shape==(24,) and np.isfinite(p).all()
print('SMOKE PASS', F.shape, p.min(), p.max())
