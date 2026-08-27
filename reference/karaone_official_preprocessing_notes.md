# Official KaraOne preprocessing notes

Source: [The KARA ONE Database](http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html), accessed during the current refinement.

The official documentation describes a 64-channel Neuroscan Quick-cap sampled at 1 kHz, seven phonemic/syllabic prompts (`iy`, `uw`, `piy`, `tiy`, `diy`, `m`, `n`), and four word prompts (`pat`, `pot`, `knew`, `gnaw`). Each trial includes a 5-second rest state, stimulus/auditory state, a 2-second articulator-positioning period, a 5-second imagined-speech state, and a speaking state.

For EEG preprocessing, the official page states that EEGLAB preprocessing was used, ocular artifacts were removed by blind source separation, the signal was filtered between 1 and 50 Hz, mean values were subtracted from each channel, and a small Laplacian filter was applied using adjacent-channel neighborhoods. It describes subject-independent leave-one-out evaluation in which each participant is tested using models trained on the other participants.

The active pipeline already uses the official `thinking_inds` intervals, 1--50 Hz filtering, CAR, ASR calibrated from clearing/rest intervals, robust normalization, and strict subject-level LOSO. The current refinement adds an optional/default fixed-montage small Laplacian after CAR, using only standard 10--20 channel geometry and no labels or held-out subject data. This is a declared preprocessing experiment, not a claim that it will necessarily increase LOSO performance.

The official page also notes that some auxiliary channels (including M1, M2, EKG, and Trigger) are generally not useful; the pipeline uses EEG picks and limits to the validated 62-channel signal representation.

## Reference

[1]: http://www.cs.toronto.edu/~complingweb/data/karaOne/karaOne.html "The KARA ONE Database: Phonological Categories in imagined and articulated speech"

