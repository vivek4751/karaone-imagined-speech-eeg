import importlib.util
import os

os.environ.pop('KARAONE_ADAPTIVE_VOWEL_MODE', None)
os.environ.pop('KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT', None)
os.environ['KARAONE_FAST_MODE'] = '1'
spec = importlib.util.spec_from_file_location('karaone_default_check', '/home/ubuntu/karaone_work/karaone_final_kaggle.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.ADAPTIVE_VOWEL_MODE == 'class_weight_fast', mod.ADAPTIVE_VOWEL_MODE
assert mod.REQUIRE_ADAPTIVE_VOWEL_CLASS_WEIGHT is True
print('default_mode=', mod.ADAPTIVE_VOWEL_MODE)

os.environ['KARAONE_ADAPTIVE_VOWEL_MODE'] = 'threshold_only_fast'
os.environ.pop('KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT', None)
spec2 = importlib.util.spec_from_file_location('karaone_override_check', '/home/ubuntu/karaone_work/karaone_final_kaggle.py')
mod2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(mod2)
assert mod2.ADAPTIVE_VOWEL_MODE == 'class_weight_fast', mod2.ADAPTIVE_VOWEL_MODE
assert mod2.ADAPTIVE_VOWEL_MODE_OVERRIDE == 'threshold_only_fast->class_weight_fast'
print('stale_threshold_mode_override=', mod2.ADAPTIVE_VOWEL_MODE_OVERRIDE)
print('SUCCESS')

os.environ.pop('KARAONE_ADAPTIVE_VOWEL_MODE', None)
os.environ.pop('KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT', None)
os.environ['KARAONE_FINE_ADAPTIVE_VOWEL_GRID'] = '1'
spec_fine = importlib.util.spec_from_file_location('karaone_fine_grid_check', '/home/ubuntu/karaone_work/karaone_final_kaggle.py')
mod_fine = importlib.util.module_from_spec(spec_fine)
spec_fine.loader.exec_module(mod_fine)
assert mod_fine.ADAPTIVE_VOWEL_WEIGHTS == [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0], mod_fine.ADAPTIVE_VOWEL_WEIGHTS
print('fine_decimal_grid=', mod_fine.ADAPTIVE_VOWEL_WEIGHTS)

os.environ.pop('KARAONE_FINE_ADAPTIVE_VOWEL_GRID', None)
os.environ['KARAONE_ADAPTIVE_VOWEL_WEIGHTS'] = '1.0,1.25,2.5,1.25'
spec_above = importlib.util.spec_from_file_location('karaone_above_two_check', '/home/ubuntu/karaone_work/karaone_final_kaggle.py')
mod_above = importlib.util.module_from_spec(spec_above)
spec_above.loader.exec_module(mod_above)
assert mod_above.ADAPTIVE_VOWEL_WEIGHTS == [1.0, 1.25, 2.5], mod_above.ADAPTIVE_VOWEL_WEIGHTS
print('explicit_decimal_above_two=', mod_above.ADAPTIVE_VOWEL_WEIGHTS)

os.environ.pop('KARAONE_ADAPTIVE_VOWEL_WEIGHTS', None)

os.environ.pop('KARAONE_ADAPTIVE_VOWEL_MODE', None)
os.environ['KARAONE_REQUIRE_VOWEL_CLASS_WEIGHT'] = '0'
spec3 = importlib.util.spec_from_file_location('karaone_optout_check', '/home/ubuntu/karaone_work/karaone_final_kaggle.py')
mod3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(mod3)
assert mod3.ADAPTIVE_VOWEL_MODE == 'class_weight_fast'
print('explicit_optout_without_threshold_request_keeps_default=', mod3.ADAPTIVE_VOWEL_MODE)
print('SUCCESS')
