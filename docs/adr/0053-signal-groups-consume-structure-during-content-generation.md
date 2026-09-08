# Signal groups consume structure during content generation

Grouped latent feature sources are implemented in the content-generation stage.
HSBM first samples foreign-key identifiers and hierarchical block paths. The
signal-group feature constructor then consumes the referenced parent rows and
block paths, together with temporal and intrinsic signals, to generate shared
feature values. The mechanism carries upstream relational structure into table
content while leaving schema, foreign-key, and timestamp sampling to their
respective upstream mechanisms.
