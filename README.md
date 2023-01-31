
# 🍌 instruct-pix2pix for banana

A working deployment of [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix) for banana.dev.

<img src='https://instruct-pix2pix.timothybrooks.com/teaser.jpg'/>

## Instant Deploy
instruct-pix2pix is available as a prebult model on Banana! [See how to deploy in seconds](https://app.banana.dev/templates/patienceai/instruct-pix2pix-banana).

## Model Inputs

The model accepts the following inputs:

* `prompt` (required)
* `image_url` (required, should be 512x512 or another standard Stable Diffusion 1.5 resolution for best results)
* `negative_prompt` (optional)
* `seed` (optional)
* `guidance_scale` (optional, default 7.5)
* `image_guidance_scale` (optional, default 1.5)

Please see the instruct-pix2pix documentation for more information about these configuration options.
