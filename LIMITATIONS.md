# Preview limitations

This is an experimental local player and research toolkit. Its controls can be useful even when detection is wrong, because the timeline can be reviewed and corrected. Unattended filtering and child-safety reliability have not been established.

The most recent development comparison used 42 human-reviewed clips. An optional crop check reduced incorrect nudity-filter time by 44.8% while preserving the measured true coverage, but nudity time precision was only 35.3% and recall only 31.9%. There were three labeled nudity events across two titles. These sparse, repeatedly inspected development examples do not establish population performance. The optional crop check is not enabled by this source release.

No category is newly qualified for automatic filtering. Kissing, sexual activity, violence and gore still have important false positives and misses. The scene classifier's automatic thresholds remain disabled. Language recognition has not been adequately evaluated on independent everyday speech. The project has not achieved 95% precision and recall.

Anatomy detection does not establish sexual activity. Ordinary kissing is a separate category. Regional boxes can miss intended pixels, lag movement or cover unrelated objects. Segmentation drafts remain research tools and are not enabled in ordinary playback. Blur preserves some visible shape; complete concealment uses an opaque picture but only during the intervals the timeline specifies.

Review-required scene suggestions need confirmation before filtering. Anatomy can filter automatically. Confirming a wrong suggestion still produces a wrong filter. The user must inspect important media and correct missed intervals; a completed scan only confirms that the configured analysis finished.

The lookahead buffer cannot compensate indefinitely for processing slower than playback. The player may pause while waiting for analysis. Speed depends on model, content, hardware, other GPU use and settings. Short sampled shots can be missed at every sampling rate. Sustained live throughput, reconnection, long recordings and broad hardware support still require testing.

Mistake reports are a selected collection, not an accuracy benchmark. A report's surrounding window is not an exact content label. Confirmed reports need independent review and precise annotation before training. Preserve title-level dataset splits and collect independently sampled evaluation data.
