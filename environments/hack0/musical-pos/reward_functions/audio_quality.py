import numpy as np
import librosa
import torch
import torchaudio
from typing import Dict, Any, Tuple, List, Optional

class AudioQualityReward:
    """
    A reward function that evaluates the technical quality of generated audio.
    
    This examines various technical aspects of audio quality including:
    1. Signal-to-noise ratio and clarity
    2. Spectral balance and frequency response
    3. Artifact detection (e.g., clipping, quantization noise)
    4. Dynamic range
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the audio quality reward function.
        
        Args:
            config: Configuration dictionary with parameters for quality assessment
        """
        self.config = config or {}
        self.sr = self.config.get("sample_rate", 44100)
        
        # Quality threshold values
        self.snr_threshold = self.config.get("snr_threshold", 20.0)  # dB
        self.clip_threshold = self.config.get("clip_threshold", 0.99)
        self.spectral_flatness_weight = self.config.get("spectral_flatness_weight", 0.3)
        self.clarity_weight = self.config.get("clarity_weight", 0.4)
        self.dynamics_weight = self.config.get("dynamics_weight", 0.2)
        self.artifact_penalty_weight = self.config.get("artifact_penalty_weight", 0.1)
        
    def estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio of audio.
        
        Args:
            audio: Audio signal (numpy array)
        
        Returns:
            Estimated SNR in dB
        """
        # Use spectral subtraction-based approach to estimate noise
        # This is a simplified estimation - in production we'd use a more sophisticated method
        
        # Calculate the spectrum
        spec = np.abs(librosa.stft(audio))
        
        # Estimate noise floor as the 5th percentile of spectral magnitudes
        noise_floor = np.percentile(spec, 5, axis=1)
        noise_floor = np.expand_dims(noise_floor, axis=1)
        
        # Estimate signal power
        signal_power = np.mean(spec**2)
        
        # Estimate noise power
        noise_power = np.mean(noise_floor**2)
        
        # Avoid division by zero
        if noise_power < 1e-10:
            return 100.0
            
        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        return min(snr_db, 100.0)  # Cap at 100 dB to avoid infinity

    def calculate_spectral_balance(self, audio: np.ndarray) -> float:
        """
        Evaluate spectral balance of audio.
        
        Args:
            audio: Audio signal (numpy array)
            
        Returns:
            Score for spectral balance (0-1)
        """
        # Calculate frequency bands energy
        spec = np.abs(librosa.stft(audio))
        
        # Divide spectrum into frequency bands
        band_edges = [0, 60, 250, 500, 2000, 4000, 10000, 20000]
        band_energies = []
        
        for i in range(len(band_edges) - 1):
            low_bin = int(band_edges[i] * spec.shape[0] / (self.sr / 2))
            high_bin = int(band_edges[i+1] * spec.shape[0] / (self.sr / 2))
            
            # Cap at max frequency bins
            high_bin = min(high_bin, spec.shape[0])
            
            if low_bin < high_bin:
                band_energy = np.mean(np.sum(spec[low_bin:high_bin, :] ** 2, axis=0))
                band_energies.append(band_energy)
            else:
                band_energies.append(0)
                
        # Normalize band energies
        if sum(band_energies) > 0:
            band_energies = [e / sum(band_energies) for e in band_energies]
        
        # Calculate spectral flatness (1 = flat, 0 = concentrated in one band)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0].mean()
        
        # Calculate spectral centroid normalized (0-1)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0].mean()
        normalized_centroid = min(centroid / (self.sr / 2), 1.0)
        
        # Check if energy is too concentrated in low or high frequencies
        low_freq_ratio = sum(band_energies[:3]) / (sum(band_energies) + 1e-8)
        high_freq_ratio = sum(band_energies[5:]) / (sum(band_energies) + 1e-8)
        
        # Penalize if too much energy is concentrated in extremes
        balance_score = 1.0 - (abs(low_freq_ratio - 0.4) + abs(high_freq_ratio - 0.3))
        
        # Combine metrics
        balance_score = max(0.0, min(1.0, balance_score))
        spectral_score = 0.5 * balance_score + 0.5 * spectral_flatness
        
        return spectral_score
        
    def detect_artifacts(self, audio: np.ndarray) -> Tuple[float, List[str]]:
        """
        Detect audio artifacts and calculate penalty.
        
        Args:
            audio: Audio signal (numpy array)
            
        Returns:
            Tuple of (artifact penalty score, list of detected artifacts)
        """
        artifacts = []
        penalty = 0.0
        
        # Check for clipping
        clip_ratio = np.mean(np.abs(audio) > self.clip_threshold)
        if clip_ratio > 0.01:
            artifacts.append(f"clipping ({clip_ratio:.2%})")
            penalty += clip_ratio * 0.5
        
        # Check for digital silence or near-silence
        if np.max(np.abs(audio)) < 0.01:
            artifacts.append("near silence")
            penalty += 0.5
            
        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.05:
            artifacts.append(f"DC offset ({dc_offset:.3f})")
            penalty += min(abs(dc_offset) * 5.0, 0.3)
            
        # Check for discontinuities (clicks/pops)
        diff = np.diff(audio)
        large_jumps = np.sum(np.abs(diff) > 0.3)
        if large_jumps > 10:
            artifacts.append(f"clicks/discontinuities ({large_jumps} detected)")
            penalty += min(large_jumps / 1000, 0.4)
        
        # Check for excessive repetition (potential looping artifacts)
        # This is a simplified check - in production we'd use more sophisticated methods
        if len(audio) > 4410:  # At least 0.1s
            corr = np.correlate(audio[:4410], audio, mode='valid')
            peaks = librosa.util.peak_pick(corr, 100, 100, 100, 100, 0.5, 10)
            if len(peaks) > 5:
                artifacts.append("excessive repetition")
                penalty += 0.3
                
        return min(penalty, 1.0), artifacts
        
    def calculate_dynamics(self, audio: np.ndarray) -> float:
        """
        Calculate dynamic range score.
        
        Args:
            audio: Audio signal (numpy array)
            
        Returns:
            Dynamic range score (0-1)
        """
        # Use librosa RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        
        # Calculate crest factor (peak to RMS ratio)
        peak = np.max(np.abs(audio))
        avg_rms = np.mean(rms)
        
        if avg_rms < 1e-6:
            return 0.0  # Effectively silent
            
        crest_factor_db = 20 * np.log10(peak / avg_rms)
        
        # Calculate dynamic range using percentiles
        if len(rms) > 10:
            p95 = np.percentile(rms, 95)
            p05 = np.percentile(rms, 5)
            
            if p05 < 1e-6:
                p05 = 1e-6
                
            dynamic_range_db = 20 * np.log10(p95 / p05)
        else:
            dynamic_range_db = 0
            
        # Score is based on having appropriate crest factor and dynamic range
        # For most music/audio, a crest factor of 10-20dB is good
        cf_score = 1.0 - min(abs(crest_factor_db - 15) / 15, 1.0)
        
        # For most styles, at least 10dB of dynamic range is desirable
        dr_score = min(dynamic_range_db / 15, 1.0)
        
        # Combine scores
        dynamics_score = 0.5 * cf_score + 0.5 * dr_score
        
        return dynamics_score
        
    def calculate_quality_score(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Calculate overall audio quality score.
        
        Args:
            audio: Audio signal (numpy array)
            
        Returns:
            Dictionary with quality scores and details
        """
        # Ensure audio is mono for analysis (average channels if stereo)
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
            
        # Calculate SNR
        snr = self.estimate_snr(audio_mono)
        snr_score = min(snr / self.snr_threshold, 1.0)
        
        # Calculate spectral balance
        spectral_score = self.calculate_spectral_balance(audio_mono)
        
        # Calculate dynamics
        dynamics_score = self.calculate_dynamics(audio_mono)
        
        # Detect artifacts
        artifact_penalty, artifacts = self.detect_artifacts(audio_mono)
        
        # Calculate clarity score (combination of SNR and artifact-free)
        clarity_score = snr_score * (1.0 - artifact_penalty)
        
        # Calculate overall quality score
        overall_score = (
            self.clarity_weight * clarity_score +
            self.spectral_flatness_weight * spectral_score +
            self.dynamics_weight * dynamics_score -
            self.artifact_penalty_weight * artifact_penalty
        )
        
        # Ensure score is in 0-1 range
        overall_score = max(0.0, min(1.0, overall_score))
        
        return {
            "overall_quality_score": overall_score,
            "clarity_score": clarity_score,
            "spectral_score": spectral_score,
            "dynamics_score": dynamics_score,
            "artifact_penalty": artifact_penalty,
            "snr_db": snr,
            "detected_artifacts": artifacts
        }
    
    def calculate_reward(self, audio_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Calculate reward from audio tensor.
        
        Args:
            audio_tensor: Audio tensor from stable-audio-open-small (shape: [channels, samples])
            
        Returns:
            Dictionary with reward score and component scores
        """
        # Convert tensor to numpy for analysis
        audio_np = audio_tensor.cpu().numpy()
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_score(audio_np)
        
        return quality_metrics