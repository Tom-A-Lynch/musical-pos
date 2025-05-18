# AudioForge Implementation Plan

## Project Structure

```
/environments/hack0/musical-pos/
├── README.md
├── requirements.txt
├── audioforge_env.py
├── data/
│   ├── target_styles.json
│   ├── prompt_templates.json
│   ├── parameter_presets.json
│   └── evaluation_references/
│       ├── drums/
│       ├── instruments/
│       ├── sfx/
│       └── ambient/
├── reward_functions/
│   ├── __init__.py
│   ├── audio_quality.py
│   ├── style_consistency.py
│   ├── prompt_alignment.py
│   ├── technical_characteristics.py
│   ├── creativity.py
│   └── combined_reward.py
├── utils/
│   ├── __init__.py
│   ├── audio_processing.py
│   ├── spectral_analysis.py
│   ├── prompt_templates.py
│   ├── parameter_optimizer.py
│   └── wandb_logger.py
└── artifacts/
    ├── audioforge_rollouts.jsonl
    └── audioforge_rollouts.html
```

## Core Components

### 1. Main Environment Class (`audioforge_env.py`)

The main environment file will subclass Atropos's base environment class and implement:

- `__init__`: Initialize the environment with configs
- `get_prompt`: Generate prompts for audio generation, including text descriptions and parameter settings
- `score_response`: Apply reward functions to generated audio outputs
- `parse_response`: Extract structured information including audio generation parameters
- `process_batch`: Handle batch processing of audio generations
- `metric_calculations`: Calculate audio quality metrics and parameter effectiveness
- Custom configuration via `config_init` class method

### 2. Data Structure

Audio generation targets in `target_styles.json` will have a format like:
```json
{
  "id": "drums-techno-128",
  "category": "drums",
  "subcategory": "techno",
  "target_description": "128 BPM techno drum loop with punchy kick and crisp hi-hats",
  "reference_file": "drums/techno_128_reference.wav",
  "parameters": {
    "seconds_total": 4.0,
    "target_bpm": 128,
    "tone_characteristics": ["punchy", "crisp", "tight"]
  },
  "baseline_prompt": "128 BPM techno drum loop with punchy kick and crisp hi-hats",
  "difficulty": "medium",
  "metadata": {
    "suitable_sampler": "pingpong",
    "recommended_steps": 8,
    "typical_cfg_scale": 7.0
  }
}
```

### 3. Reward Functions

We'll implement multiple reward components:

#### a. Audio Quality Assessment (`audio_quality.py`)
- Signal-to-noise ratio analysis
- Artifact detection
- Overall clarity metrics
- Frequency balance metrics

#### b. Style Consistency (`style_consistency.py`)
- Spectral similarity to reference examples
- Genre/style characteristic detection
- BPM/rhythm accuracy (for rhythmic content)

#### c. Prompt Alignment (`prompt_alignment.py`)
- Semantic matching between prompt and audio characteristics
- Presence of requested elements
- Adherence to specified audio characteristics

#### d. Technical Characteristics (`technical_characteristics.py`)
- Rhythm accuracy and consistency
- Stereo imaging quality
- Frequency spectrum balance
- Dynamic range appropriate to style

#### e. Creativity/Novelty (`creativity.py`)
- Uniqueness compared to training examples
- Interesting/surprising elements
- Coherent but novel combinations

#### f. Combined Reward Function (`combined_reward.py`)
- Weighted combination of individual rewards
- Configuration options for category-specific weightings

### 4. Utility Modules

#### a. Audio Processing (`audio_processing.py`)
- Audio loading and saving
- Format conversion
- Normalization
- Feature extraction

#### b. Spectral Analysis (`spectral_analysis.py`)
- FFT and spectrogram generation
- Frequency content analysis
- Rhythm and beat detection
- Timbral characteristic extraction

#### c. Prompt Templates (`prompt_templates.py`)
- Category-specific prompt formats
- Parameter recommendation systems
- Few-shot examples for different audio types

#### d. Parameter Optimizer (`parameter_optimizer.py`)
- Diffusion parameter management
- Optimal parameter search
- Setting recommendation engine

#### e. WandB Logger (`wandb_logger.py`)
- Audio sample logging
- Spectral visualization
- Parameter tracking
- A/B comparison tools

## Implementation Phases

### Phase 1: Core Infrastructure
- Environment class setup
- Basic stable-audio-open-small integration
- Simple audio quality metrics
- Test pipeline with drum loop generation

### Phase 2: Enhanced Evaluation
- Implement all reward components
- Create comprehensive spectral metrics
- Expand to multiple audio categories
- Parameter optimization logic

### Phase 3: Optimization & Analysis
- Fine-tune reward weights by category
- Add visualization components
- Create evaluation dashboard
- Generate sample rollouts for hackathon demo

## Requirements

Main dependencies will include:
- atroposlib
- stable-audio-tools
- torch
- torchaudio
- librosa (for audio analysis)
- numpy
- scipy
- wandb
- matplotlib (for visualization)

## Integration with Atropos

The environment will follow standard Atropos integration:
1. Compatible with the API service via `serve` mode
2. Support for local testing via `process` mode
3. Generate artifacts for submission
4. Track metrics through WandB for analysis

## Integration with stable-audio-open-small

Key integration points:
1. Model loading and setup:
```python
from stable_audio_tools import get_pretrained_model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-small")
```

2. Audio generation with the model:
```python
from stable_audio_tools.inference.generation import generate_diffusion_cond

output = generate_diffusion_cond(
    model, 
    steps=8, 
    conditioning=[{
        "prompt": prompt,
        "seconds_total": audio_length
    }],
    sample_size=model_config["sample_size"],
    sampler_type="pingpong",
    device=device
)
```

3. Parameter optimization:
```python
# Example parameter ranges to explore
step_options = [4, 8, 12, 16]
cfg_scales = [3.0, 5.0, 7.0, 9.0]
sampler_types = ["pingpong", "dpmpp-sde"]

# The environment will explore combinations and learn optimal settings
```

## Evaluation Strategy

We'll evaluate generations on:
1. Overall audio quality (clarity, artifacts, balance)
2. Style matching to target category
3. Prompt following ability
4. Technical sound quality
5. Creativity and interestingness

Each metric will be normalized to a 0-1 scale for consistent reporting.

## LLM Training Focus

The RL environment will focus on training LLMs to:
1. Generate better text prompts for audio synthesis
2. Recommend optimal diffusion parameters for specific audio goals
3. Structure effective instructions for audio generation
4. Learn the capabilities and limitations of the audio model

This will create a system where an LLM can act as an intelligent interface to the audio generation model, significantly improving output quality through optimized prompting and parameter selection.