A permutation assessment no longer fails when `run()` was given neither `groups` nor `sample_metadata`: every permutation re-ran the experiment with an empty metadata table, which `run()` rejected.
