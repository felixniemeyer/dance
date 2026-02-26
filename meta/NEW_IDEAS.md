# New Ideas

## Phase rate output
Models output `[sin(phase), cos(phase), phase_rate]` where `phase_rate` is radians per frame.
The application layer can apply any time offset: `phase += phase_rate * offset_frames`.
This keeps the model simple (predict current phase only) while enabling flexible lookahead at inference time.
