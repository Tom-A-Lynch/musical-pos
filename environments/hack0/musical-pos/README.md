# 🎵 AudioForge RL Environment

> An Atropos reinforcement learning environment for tuning text-to-audio generation models with Stable Audio

## Overview

AudioForge is an Atropos RL environment designed to improve text-to-audio generation capabilities using Stability AI's `stable-audio-open-small` model. This environment enables training LLMs to generate better text prompts for audio generation, fine-tuning the diffusion process parameters, and optimizing the quality of generated audio across various sound categories.

This project was created for the Nous Research RL Environments Hackathon on May 18, 2025.

## Motivation

Text-to-audio generation is a rapidly emerging field with significant potential for creative applications. However, current models like `stable-audio-open-small` have several limitations:

1. Highly variable quality depending on prompt engineering
2. Inconsistent performance across different audio styles and categories
3. Limited ability to generate certain types of sounds, especially vocals
4. Difficulty in finding optimal diffusion parameters for specific audio goals

AudioForge addresses these challenges by creating a reinforcement learning environment that rewards:
- Clear, high-quality audio generation
- Effective prompt engineering for specific audio targets
- Optimal parameter selection for the diffusion process
- Style-consistent audio generation across various categories

This environment falls under the **Subjective Track** as it focuses on artistic, creative audio generation that enhances the model's capabilities in producing engaging and high-quality audio outputs.

## Environment Design

The AudioForge environment leverages `stable-audio-open-small` to generate audio samples based on text prompts and diffusion parameters, then evaluates them using a multi-component reward system.

### Audio Categories
- Drum Loops (various BPM and styles)
- Instrument Samples (piano, guitar, synth, etc.)
- Sound Effects (nature, mechanical, abstract)
- Ambient Textures (atmospheric, textural)
- Musical Phrases (melodic patterns, bass lines)

### Reward Function Components
1. **Audio Quality Assessment**: Evaluates clarity, absence of artifacts, and overall fidelity
2. **Style Consistency**: Measures how well the generated audio matches the intended style
3. **Prompt Alignment**: Assesses how closely the audio follows the text prompt
4. **Technical Characteristics**: Analyzes rhythm accuracy, frequency balance, stereo imaging
5. **Novelty/Creativity**: Rewards unique and interesting audio generations

### Implementation Architecture
- Prompt Engineering Module: Optimizes text descriptions for audio generation
- Parameter Optimization System: Finds ideal diffusion settings for specific audio goals
- Audio Quality Evaluation Pipeline: Multi-dimensional scoring of generated samples
- Results Visualization Dashboard: Tracks improvements across different audio categories

## Quick Start

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the environment in process mode to test:
```bash
python audioforge_env.py process --openai.model_name your_model_name --env.data_path_to_save_groups audioforge_rollouts.jsonl
```

3. Run as a service to integrate with Atropos:
```bash
python audioforge_env.py serve --openai.model_name your_model_name --slurm false
```

## WandB Integration

Our environment logs detailed metrics to Weights & Biases to track model performance across different audio categories and parameters.

[View our sample WandB run](https://wandb.ai/sample/audioforge/runs/sample123)

The dashboard includes:
- Audio quality scores by category
- Spectral analysis visualizations
- Parameter optimization tracking
- A/B comparisons of generations

## Technical Implementation

AudioForge uses the `stable-audio-tools` library to interface with the `stable-audio-open-small` model, which enables:

- Variable-length stereo audio generation (up to 11 seconds) at 44.1kHz
- Text prompt conditioning with T5-based text embeddings
- Transformer-based diffusion (DiT) in the latent space
- On-device generation optimized for Arm CPUs

Key model parameters we optimize include:
- Number of diffusion steps
- Sampler type selection
- Classifier-free guidance scale
- Sigma min/max values
- Timing and length parameters

## Future Improvements

- Add multi-modal reward systems using audio analysis models
- Implement comparative evaluation against reference audio datasets
- Support longer audio generation through segment stitching
- Create specialized training for genre-specific audio generation
- Develop interactive tools for exploring the learned audio latent space

## Team

- [Team Member 1] - Audio engineering specialist
- [Team Member 2] - Machine learning engineer
- [Team Member 3] - RL environment architect
- [Team Member 4] - Creative audio producer

## License

MIT

---

## Extended Environment Description

The AudioForge environment expands beyond basic audio generation to support creative applications including:

1. **Theme-based Generation**: Creating audio that matches specific moods, environments or scenarios
2. **Parameter Optimization**: Finding ideal diffusion settings for each audio category
3. **Style Transfer**: Adapting the characteristics of one sound to another context
4. **Multi-prompt Composition**: Building complex audio scenes from multiple prompt components

Our goal is to develop a framework that elevates text-to-audio generation from simple proof-of-concept to a reliable creative tool for musicians, sound designers, and media producers.