`BIDSDataset` keeps MNE and mne-bids quiet for the whole load: per-call `verbose` arguments changed the process-wide log level while other threads were reading, which let warnings through at random.
