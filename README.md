## Cog implementation of stable audio

Stable Audio Open is an open-source model optimized for generating short audio samples, sound effects, and production elements using text prompts. Ideal for creating drum beats, instrument riffs, ambient sounds, foley recordings, and other audio samples, the model was trained on data from Freesound and the Free Music Archive, respecting creator rights.

For more information about this model, see [here](https://stability.ai/news/introducing-stable-audio-open).

You can demo this model or learn how to use it with Replicate's API [here](https://replicate.com/stackadoc/stable-audio-open-1.0). 

## Run with Cog

Cog is an open-source tool that packages machine learning models in a standard, production-ready container. You can deploy your packaged model to your own infrastructure, or to Replicate, where users can interact with it via web interface or API.
Prerequisites

Cog. Follow these instructions to install Cog, or just run:

```bash
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```

Note, to use Cog, you'll also need an installation of Docker.

GPU machine. You'll need a Linux machine with an NVIDIA GPU attached and the NVIDIA Container Toolkit installed. If you don't already have access to a machine with a GPU, check out our guide to getting a GPU machine.

### Step 1. Clone this repository

```bash
git clone https://github.com/stackadoc/cog-stable-audio.git
```

### Step 2. Run the model

To run the model, you need a local copy of the model's Docker image. You can satisfy this requirement by specifying the image ID in your call to predict like:

```bash
cog predict r8.im/stackadoc/stable-audio-open-1.0 -i prompt=tense staccato strings. plucked strings. dissonant. scary movie. -i duration=8
```

For more information, see the Cog section here

Alternatively, you can build the image yourself, either by running cog build or by letting cog predict trigger the build process implicitly. For example, the following will trigger the build process and then execute prediction:

```bash
cog predict -i description="tense staccato strings. plucked strings. dissonant. scary movie." -i duration=8
```

Note, the first time you run cog predict, model weights and other requisite assets will be downloaded if they're not available locally. This download only needs to be executed once.

## Run on replicate

### Step 1. Ensure that all assets are available locally

If you haven't already, you should ensure that your model runs locally with cog predict. This will guarantee that all assets are accessible. E.g., run:

cog predict -i description=tense staccato strings. plucked strings. dissonant. scary movie. -i duration=8

### Step 2. Create a model on Replicate.

Go to replicate.com/create to create a Replicate model. If you want to keep the model private, make sure to specify "private".

### Step 3. Configure the model's hardware

Replicate supports running models on variety of CPU and GPU configurations. For the best performance, you'll want to run this model on an A100 instance.

Click on the "Settings" tab on your model page, scroll down to "GPU hardware", and select "A100". Then click "Save".

### Step 4: Push the model to Replicate

Log in to Replicate:

```bash
cog login
```

Push the contents of your current directory to Replicate, using the model name you specified in step 1:

```bash
cog push r8.im/username/modelname
```

Learn more about pushing models to Replicate.
