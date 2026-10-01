Changelog
=========

Upcoming version (not yet released)
----------------------------------

Added
^^^^^

- Added offline vertex-ray LP contact-margin labeling, physical augmentation,
  controller/episode-held-out MLP training and frozen export. The optional
  control-rate surrogate reward is disabled by default; existing simulation,
  rewards, curriculum and terminations are unchanged. Actual contact tangents
  extend the requested 35 raw features to 37 for the pyramidal friction model.
