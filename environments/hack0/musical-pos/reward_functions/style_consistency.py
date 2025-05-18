import numpy as np
import librosa
import torch
import torchaudio
from typing import Dict, Any, List, Optional, Tuple
import os

class StyleConsistencyReward:
    """
    A reward function that evaluates how well the generated audio matches a target style.
    
    This compares various aspects of the generated audio against style references or targets:
    1. Spectral similarity to reference examples
    2. BPM/rhythm accuracy (for rhythmic content)
    3. Genre/style characteristic detection
    4. Adherence to requested musical parameters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the style consistency reward function.
        
        Args:
            config: Configuration dictionary with parameters for style evaluation
        """
        self.config = config or {}
        self.sr = self.config.get("sample_rate", 44100)
        self.reference_dir = self.config.get("reference_dir", "data/evaluation_references")
        self.spectral_weight = self.config.get("spectral_weight", 0.4)
        self.rhythm_weight = self.config.get("rhythm_weight", 0.3)
        self.instrument_weight = self.config.get("instrument_weight", 0.3)
        
        # Cache for reference audio features
        self._reference_features = {}
        
    def load_reference_features(self, category: str, subcategory: str) -> Dict[str, Any]:
        """
        Load or compute reference features for a specific style.
        
        Args:
            category: Main category (e.g., "drums", "ambient")
            subcategory: Subcategory or style (e.g., "techno", "ambient_dark")
            
        Returns:
            Dictionary of reference features
        """
        cache_key = f"{category}/{subcategory}"
        
        # Return cached features if available
        if cache_key in self._reference_features:
            return self._reference_features[cache_key]
            
        # Build path to reference files
        ref_path = os.path.join(self.reference_dir, category)
        
        # Find matching reference files
        reference_files = []
        if os.path.exists(ref_path):
            for file in os.listdir(ref_path):
                if file.startswith(f"{subcategory}_") and file.endswith((".wav", ".mp3")):
                    reference_files.append(os.path.join(ref_path, file))
        
        # No reference files found
        if not reference_files:
            return {}
            
        # Load and compute features for references
        mfccs_list = []
        chroma_list = []
        onsets_list = []
        tempos = []
        
        for ref_file in reference_files:
            try:
                # Load audio
                audio, sr = librosa.load(ref_file, sr=self.sr, mono=True)
                
                # Extract MFCCs
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfccs_list.append(mfcc)
                
                # Extract chroma
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
                chroma_list.append(chroma)
                
                # Detect onsets
                onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
                onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
                onsets_list.append(onset_env)
                
                # Estimate tempo
                tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                tempos.append(tempo)
                
            except Exception as e:
                print(f"Error processing reference file {ref_file}: {e}")
                continue
        
        # Average features across references
        if mfccs_list:
            avg_mfcc = np.mean(np.array(mfccs_list), axis=0)
            avg_chroma = np.mean(np.array(chroma_list), axis=0)
            avg_tempo = np.mean(tempos)
            
            features = {
                "mfcc": avg_mfcc,
                "chroma": avg_chroma,
                "tempo": avg_tempo,
                "onsets": onsets_list[0] if onsets_list else None  # Use first one as reference
            }
            
            # Cache the computed features
            self._reference_features[cache_key] = features
            
            return features
        
        return {}
        
    def calculate_spectral_similarity(self, audio: np.ndarray, reference_features: Dict[str, Any]) -> float:
        """
        Calculate spectral similarity between generated audio and reference.
        
        Args:
            audio: Generated audio array
            reference_features: Dictionary of reference audio features
            
        Returns:
            Similarity score (0-1)
        """
        if not reference_features or "mfcc" not in reference_features:
            return 0.5  # No reference to compare against
            
        # Extract MFCCs from generated audio
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # Calculate distance between MFCCs
        # Dynamic time warping would be better but more computationally expensive
        # We use a simpler approach here
        
        # Adjust lengths to match for comparison
        ref_mfcc = reference_features["mfcc"]
        if mfcc.shape[1] > ref_mfcc.shape[1]:
            mfcc = mfcc[:, :ref_mfcc.shape[1]]
        elif mfcc.shape[1] < ref_mfcc.shape[1]:
            ref_mfcc = ref_mfcc[:, :mfcc.shape[1]]
            
        # Calculate normalized Euclidean distance
        mfcc_dist = np.mean(np.sqrt(np.sum((mfcc - ref_mfcc)**2, axis=0)))
        max_dist = np.sqrt(13 * 4)  # Rough max distance for MFCCs
        mfcc_sim = 1.0 - min(mfcc_dist / max_dist, 1.0)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
        ref_chroma = reference_features.get("chroma")
        
        if ref_chroma is not None:
            # Adjust lengths to match
            if chroma.shape[1] > ref_chroma.shape[1]:
                chroma = chroma[:, :ref_chroma.shape[1]]
            elif chroma.shape[1] < ref_chroma.shape[1]:
                ref_chroma = ref_chroma[:, :chroma.shape[1]]
                
            # Calculate chroma similarity (cosine similarity)
            chroma_flat = chroma.flatten()
            ref_chroma_flat = ref_chroma.flatten()
            
            chroma_norm = np.linalg.norm(chroma_flat)
            ref_chroma_norm = np.linalg.norm(ref_chroma_flat)
            
            if chroma_norm > 0 and ref_chroma_norm > 0:
                chroma_sim = np.dot(chroma_flat, ref_chroma_flat) / (chroma_norm * ref_chroma_norm)
                chroma_sim = (chroma_sim + 1) / 2  # Convert from [-1,1] to [0,1]
            else:
                chroma_sim = 0.5
        else:
            chroma_sim = 0.5
            
        # Combine similarities
        spectral_sim = 0.6 * mfcc_sim + 0.4 * chroma_sim
        
        return spectral_sim
        
    def calculate_rhythm_accuracy(self, audio: np.ndarray, reference_features: Dict[str, Any], target_bpm: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate rhythm and tempo accuracy.
        
        Args:
            audio: Generated audio array
            reference_features: Dictionary of reference audio features
            target_bpm: Target BPM if specified in prompt
            
        Returns:
            Dictionary with rhythm accuracy metrics
        """
        # Extract rhythm features
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # Get reference tempo
        ref_tempo = reference_features.get("tempo", target_bpm)
        
        if ref_tempo is None:
            tempo_score = 0.5  # No reference
        else:
            # Calculate tempo accuracy (allow for half-time, double-time interpretations)
            tempo_ratios = [tempo / ref_tempo, (tempo * 2) / ref_tempo, tempo / (ref_tempo * 2)]
            tempo_errors = [abs(1 - ratio) for ratio in tempo_ratios]
            min_error = min(tempo_errors)
            
            # Convert error to score (0-1)
            tempo_score = max(0.0, 1.0 - min_error * 2)
        
        # Calculate rhythm pattern similarity if reference onsets available
        ref_onsets = reference_features.get("onsets")
        if ref_onsets is not None:
            # Adjust lengths to match
            if len(onset_env) > len(ref_onsets):
                onset_env = onset_env[:len(ref_onsets)]
            elif len(onset_env) < len(ref_onsets):
                ref_onsets = ref_onsets[:len(onset_env)]
                
            # Calculate correlation between onset patterns
            correlation = np.corrcoef(onset_env, ref_onsets)[0, 1]
            onset_sim = (correlation + 1) / 2  # Convert from [-1,1] to [0,1]
        else:
            onset_sim = 0.5  # No reference
            
        # Detect if rhythm is consistent
        if len(onset_env) > 0:
            # Calculate autocorrelation to detect rhythmic regularity
            acorr = librosa.autocorrelate(onset_env)
            peaks = librosa.util.peak_pick(acorr, 3, 3, 3, 3, 0.5, 10)
            
            if len(peaks) > 3:
                regularity = min(1.0, len(peaks) / 10)
            else:
                regularity = 0.3
        else:
            regularity = 0.0
            
        # Combine rhythm metrics
        rhythm_score = 0.4 * tempo_score + 0.4 * onset_sim + 0.2 * regularity
        
        return {
            "rhythm_score": rhythm_score,
            "tempo_score": tempo_score,
            "onset_similarity": onset_sim,
            "regularity": regularity,
            "detected_tempo": tempo
        }
        
    def identify_instruments(self, audio: np.ndarray, category: str) -> Dict[str, float]:
        """
        Attempt to identify key instruments or sound characteristics.
        
        Args:
            audio: Generated audio array
            category: Target category
            
        Returns:
            Dictionary with presence probabilities for key sound types
        """
        # This is a simplified placeholder - in production we would use:
        # 1. A trained classifier for instrument detection
        # 2. Source separation and analysis
        # 3. More sophisticated spectral signature matching
        
        # Extract features useful for instrument identification
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0].mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0].mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0].mean()
        
        # Normalize to 0-1 range
        norm_centroid = min(spectral_centroid / (self.sr / 2), 1.0)
        norm_bandwidth = min(spectral_bandwidth / (self.sr / 4), 1.0)
        norm_rolloff = min(spectral_rolloff / (self.sr / 2), 1.0)
        
        # Different categories have different expected spectral characteristics
        if category == "drums":
            # Drums typically have high zero-crossing rate, varied centroid
            kick_presence = 1.0 - norm_centroid  # Kick has low centroid
            snare_presence = zero_crossing_rate * 10  # Snare has high ZCR
            hihat_presence = norm_rolloff  # Hi-hats have high rolloff
            
            return {
                "kick_drum": min(kick_presence, 1.0),
                "snare_drum": min(snare_presence, 1.0),
                "hihat": min(hihat_presence, 1.0)
            }
            
        elif category == "instruments":
            # Simplified instrument detection based on spectral properties
            # Low centroid and bandwidth -> bass
            # Moderate centroid and low bandwidth -> guitar
            # High centroid, high bandwidth -> piano/keys
            bass_presence = 1.0 - norm_centroid
            guitar_presence = 1.0 - abs(norm_centroid - 0.5) - norm_bandwidth
            keys_presence = norm_centroid * norm_bandwidth
            
            return {
                "bass": min(bass_presence, 1.0),
                "guitar": min(guitar_presence, 1.0),
                "keys": min(keys_presence, 1.0)
            }
            
        elif category == "ambient":
            # For ambient, look at characteristics like continuity
            # Calculate frame-to-frame RMS difference (lower = more continuous)
            rms = librosa.feature.rms(y=audio)[0]
            rms_diff = np.mean(np.abs(np.diff(rms)))
            continuity = 1.0 - min(rms_diff * 50, 1.0)
            
            return {
                "continuity": continuity,
                "brightness": norm_centroid,
                "complexity": norm_bandwidth
            }
        
        # Default case for other categories
        return {
            "brightness": norm_centroid,
            "richness": norm_bandwidth,
            "sharpness": norm_rolloff
        }
        
    def calculate_style_consistency(
        self, 
        audio: np.ndarray, 
        category: str, 
        subcategory: str, 
        target_bpm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate overall style consistency score.
        
        Args:
            audio: Generated audio array
            category: Target category
            subcategory: Target subcategory/style
            target_bpm: Target BPM if specified
            
        Returns:
            Dictionary with style consistency scores and details
        """
        # Ensure audio is mono for analysis (average channels if stereo)
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            audio_mono = np.mean(audio, axis=0)
        else:
            audio_mono = audio
            
        # Load reference features
        reference_features = self.load_reference_features(category, subcategory)
        
        # Calculate spectral similarity
        spectral_similarity = self.calculate_spectral_similarity(audio_mono, reference_features)
        
        # Calculate rhythm accuracy
        rhythm_metrics = self.calculate_rhythm_accuracy(audio_mono, reference_features, target_bpm)
        
        # Identify instruments/sonic characteristics
        instrument_metrics = self.identify_instruments(audio_mono, category)
        
        # Calculate instrument match score
        # Ideally this would compare to expected instruments for the style
        # Here we use a simplified approach
        instrument_score = sum(instrument_metrics.values()) / len(instrument_metrics)
        
        # Calculate overall style consistency score
        overall_score = (
            self.spectral_weight * spectral_similarity +
            self.rhythm_weight * rhythm_metrics["rhythm_score"] +
            self.instrument_weight * instrument_score
        )
        
        # Ensure score is in 0-1 range
        overall_score = max(0.0, min(1.0, overall_score))
        
        return {
            "overall_style_score": overall_score,
            "spectral_similarity": spectral_similarity,
            "rhythm_metrics": rhythm_metrics,
            "instrument_metrics": instrument_metrics,
            "reference_features_available": len(reference_features) > 0
        }
    
    def calculate_reward(
        self, 
        audio_tensor: torch.Tensor, 
        category: str, 
        subcategory: str, 
        target_bpm: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate reward from audio tensor.
        
        Args:
            audio_tensor: Audio tensor from stable-audio-open-small (shape: [channels, samples])
            category: Target category
            subcategory: Target subcategory/style
            target_bpm: Target BPM if specified
            
        Returns:
            Dictionary with reward score and component scores
        """
        # Convert tensor to numpy for analysis
        audio_np = audio_tensor.cpu().numpy()
        
        # Calculate style consistency metrics
        style_metrics = self.calculate_style_consistency(audio_np, category, subcategory, target_bpm)
        
        return style_metrics