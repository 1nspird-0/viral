#!/usr/bin/env python3
"""Prepare real health datasets for ViralFlip training.

This script:
1. Discovers downloaded health datasets
2. Extracts audio features (MFCCs, spectral features)
3. Creates train/val/test splits
4. Saves processed data in training-ready format

Usage:
    python scripts/prepare_real_data.py --data-dir data/ --output data/processed/
    python scripts/prepare_real_data.py --parallel 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from viralflip.model.virus_types import VirusType, VIRUS_NAMES, NUM_VIRUS_CLASSES


def check_dependencies():
    """Check required dependencies."""
    missing = []
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import soundfile
    except ImportError:
        missing.append("soundfile")
    
    if missing:
        print(f"Installing missing dependencies: {missing}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing + ["-q"])


# Label mapping to VirusType
LABEL_MAP = {
    # COVID
    "covid": VirusType.COVID,
    "covid-19": VirusType.COVID,
    "covid_positive": VirusType.COVID,
    "positive": VirusType.COVID,
    "sars-cov-2": VirusType.COVID,
    "symptomatic_covid": VirusType.COVID,
    
    # Flu
    "flu": VirusType.FLU,
    "influenza": VirusType.FLU,
    "ili": VirusType.FLU,
    
    # Cold
    "cold": VirusType.COLD,
    "common_cold": VirusType.COLD,
    
    # RSV
    "rsv": VirusType.RSV,
    
    # Pneumonia
    "pneumonia": VirusType.PNEUMONIA,
    
    # General respiratory
    "respiratory": VirusType.GENERAL,
    "symptomatic": VirusType.GENERAL,
    "sick": VirusType.GENERAL,
    "other": VirusType.GENERAL,
    
    # Healthy
    "healthy": VirusType.HEALTHY,
    "negative": VirusType.HEALTHY,
    "covid_negative": VirusType.HEALTHY,
    "control": VirusType.HEALTHY,
    "asymptomatic": VirusType.HEALTHY,
    "no_resp_illness_exposed": VirusType.HEALTHY,
}


def extract_audio_features(
    audio_path: str,
    sr: int = 16000,
    n_mfcc: int = 13,
) -> Optional[np.ndarray]:
    """Extract audio features from a file.
    
    Returns a 30-dimensional feature vector:
    - 13 MFCC means
    - 13 MFCC stds  
    - 4 spectral features (centroid, rolloff, zcr, rms)
    """
    try:
        import librosa
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load audio
            y, _ = librosa.load(audio_path, sr=sr, duration=10.0)
            
            if len(y) < sr * 0.3:  # Less than 0.3 seconds
                return None
            
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y))
            rms = np.mean(librosa.feature.rms(y=y))
            
            # Normalize
            spectral_centroid = spectral_centroid / 10000  # Scale to ~[0,1]
            spectral_rolloff = spectral_rolloff / 10000
            
            # Combine
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                [spectral_centroid, spectral_rolloff, zero_crossing, rms],
            ])
            
            return features.astype(np.float32)
            
    except Exception as e:
        return None


def load_coughvid(data_dir: Path) -> List[Dict]:
    """Load COUGHVID dataset."""
    samples = []
    
    # Find COUGHVID directory
    coughvid_dirs = list(data_dir.glob("*oughvid*")) + list(data_dir.glob("*ublic_dataset*"))
    if not coughvid_dirs:
        return samples
    
    coughvid_path = coughvid_dirs[0]
    print(f"Loading COUGHVID from {coughvid_path}")
    
    # Find metadata
    metadata_files = list(coughvid_path.glob("**/metadata*.csv"))
    if not metadata_files:
        return samples
    
    try:
        import pandas as pd
        df = pd.read_csv(metadata_files[0])
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="COUGHVID"):
            # Get status
            status = str(row.get("status", row.get("covid_status", "unknown"))).lower().strip()
            virus_type = LABEL_MAP.get(status, VirusType.GENERAL)
            
            # Find audio file
            uuid = str(row.get("uuid", row.get("id", "")))
            if not uuid:
                continue
            
            audio_path = None
            for ext in [".webm", ".ogg", ".wav", ".mp3"]:
                candidate = coughvid_path / f"{uuid}{ext}"
                if candidate.exists():
                    audio_path = str(candidate)
                    break
            
            if not audio_path:
                # Search in subdirs
                matches = list(coughvid_path.glob(f"**/{uuid}.*"))
                if matches:
                    audio_path = str(matches[0])
            
            quality = float(row.get("cough_detected", row.get("quality", 0.5)))
            
            samples.append({
                "sample_id": uuid,
                "dataset": "coughvid",
                "virus_type": virus_type.value,
                "audio_path": audio_path,
                "quality": quality,
            })
    except Exception as e:
        print(f"Error loading COUGHVID: {e}")
    
    return samples


def load_coswara(data_dir: Path) -> List[Dict]:
    """Load Coswara dataset."""
    samples = []
    
    # Find Coswara directory
    coswara_dirs = list(data_dir.glob("*oswara*"))
    if not coswara_dirs:
        return samples
    
    coswara_path = coswara_dirs[0]
    print(f"Loading Coswara from {coswara_path}")
    
    # Try to find metadata CSV first
    csv_files = list(coswara_path.glob("**/combined*.csv")) + list(coswara_path.glob("**/metadata*.csv"))
    
    if csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Coswara"):
                status = str(row.get("covid_status", row.get("status", "unknown"))).lower().strip()
                virus_type = LABEL_MAP.get(status, VirusType.GENERAL)
                
                samples.append({
                    "sample_id": str(row.get("id", row.name)),
                    "dataset": "coswara",
                    "virus_type": virus_type.value,
                    "audio_path": None,
                    "quality": 0.8,
                })
        except Exception as e:
            print(f"Error loading Coswara CSV: {e}")
    
    # Also check folder structure
    for user_dir in coswara_path.glob("*/"):
        if not user_dir.is_dir():
            continue
        
        # Look for any audio files
        audio_files = list(user_dir.glob("*.wav")) + list(user_dir.glob("*.mp3"))
        
        # Try to determine status from folder name or metadata
        status = "unknown"
        meta_file = user_dir / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    meta = json.load(f)
                status = str(meta.get("covid_status", "unknown")).lower()
            except:
                pass
        elif "positive" in user_dir.name.lower():
            status = "positive"
        elif "negative" in user_dir.name.lower():
            status = "negative"
        
        virus_type = LABEL_MAP.get(status, VirusType.GENERAL)
        
        for audio_file in audio_files:
            samples.append({
                "sample_id": f"{user_dir.name}_{audio_file.stem}",
                "dataset": "coswara",
                "virus_type": virus_type.value,
                "audio_path": str(audio_file),
                "quality": 0.8,
            })
    
    return samples


def load_virufy(data_dir: Path) -> List[Dict]:
    """Load Virufy dataset."""
    samples = []
    
    virufy_dirs = list(data_dir.glob("*irufy*"))
    if not virufy_dirs:
        return samples
    
    virufy_path = virufy_dirs[0]
    print(f"Loading Virufy from {virufy_path}")
    
    for audio_file in tqdm(list(virufy_path.glob("**/*.wav")), desc="Virufy"):
        # Determine label from path
        path_str = str(audio_file).lower()
        
        if "positive" in path_str or "covid" in path_str:
            virus_type = VirusType.COVID
        elif "negative" in path_str or "healthy" in path_str:
            virus_type = VirusType.HEALTHY
        else:
            virus_type = VirusType.GENERAL
        
        samples.append({
            "sample_id": audio_file.stem,
            "dataset": "virufy",
            "virus_type": virus_type.value,
            "audio_path": str(audio_file),
            "quality": 1.0,
        })
    
    return samples


def load_dicova(data_dir: Path) -> List[Dict]:
    """Load DiCOVA dataset."""
    samples = []
    
    dicova_dirs = list(data_dir.glob("*icova*")) + list(data_dir.glob("*iCOVA*"))
    if not dicova_dirs:
        return samples
    
    dicova_path = dicova_dirs[0]
    print(f"Loading DiCOVA from {dicova_path}")
    
    # Look for label files
    label_files = list(dicova_path.glob("**/labels*.csv")) + list(dicova_path.glob("**/metadata*.csv"))
    
    if label_files:
        try:
            import pandas as pd
            df = pd.read_csv(label_files[0])
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="DiCOVA"):
                label = str(row.get("label", row.get("covid_status", "unknown"))).lower().strip()
                virus_type = LABEL_MAP.get(label, VirusType.GENERAL)
                
                samples.append({
                    "sample_id": str(row.get("file_name", row.name)),
                    "dataset": "dicova",
                    "virus_type": virus_type.value,
                    "audio_path": None,
                    "quality": 0.9,
                })
        except Exception as e:
            print(f"Error loading DiCOVA: {e}")
    
    return samples


def load_flusense(data_dir: Path) -> List[Dict]:
    """Load FluSense dataset."""
    samples = []
    
    flusense_dirs = list(data_dir.glob("*lusense*")) + list(data_dir.glob("*luSense*"))
    if not flusense_dirs:
        return samples
    
    flusense_path = flusense_dirs[0]
    print(f"Loading FluSense from {flusense_path}")
    
    # FluSense structure varies - look for any annotation files
    json_files = list(flusense_path.glob("**/*.json"))
    
    for jf in json_files[:5]:  # Limit for speed
        try:
            with open(jf) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    flu_positive = item.get("flu_positive", item.get("label", False))
                    virus_type = VirusType.FLU if flu_positive else VirusType.HEALTHY
                    
                    samples.append({
                        "sample_id": str(item.get("id", len(samples))),
                        "dataset": "flusense",
                        "virus_type": virus_type.value,
                        "audio_path": None,
                        "quality": 0.7,
                    })
        except:
            continue
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare real health data for training")
    parser.add_argument("--data-dir", "-d", type=str, default="data/",
                       help="Directory containing downloaded datasets")
    parser.add_argument("--output", "-o", type=str, default="data/processed/",
                       help="Output directory for processed data")
    parser.add_argument("--parallel", "-p", type=int, default=4,
                       help="Number of parallel workers for feature extraction")
    parser.add_argument("--skip-features", action="store_true",
                       help="Skip audio feature extraction")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (for testing)")
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ViralFlip Real Data Preparation")
    print(f"{'='*60}\n")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.parallel}")
    print()
    
    # Load all datasets
    all_samples = []
    
    print("Loading datasets...")
    loaders = [
        ("COUGHVID", load_coughvid),
        ("Coswara", load_coswara),
        ("Virufy", load_virufy),
        ("DiCOVA", load_dicova),
        ("FluSense", load_flusense),
    ]
    
    for name, loader in loaders:
        try:
            samples = loader(data_dir)
            if args.max_samples:
                samples = samples[:args.max_samples]
            all_samples.extend(samples)
            print(f"  {name}: {len(samples)} samples")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    if not all_samples:
        print("\nNo samples found!")
        print("Please download health datasets first:")
        print("  python scripts/download_more_data.py --health --parallel 4")
        return
    
    print(f"\nTotal samples: {len(all_samples)}")
    
    # Extract audio features
    if not args.skip_features:
        print("\nExtracting audio features...")
        
        samples_with_audio = [s for s in all_samples if s.get("audio_path")]
        print(f"  Samples with audio: {len(samples_with_audio)}")
        
        if samples_with_audio:
            def process_sample(sample):
                audio_path = sample.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    features = extract_audio_features(audio_path)
                    if features is not None:
                        sample["audio_features"] = features.tolist()
                return sample
            
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = [executor.submit(process_sample, s) for s in samples_with_audio]
                
                for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Features")):
                    try:
                        result = future.result()
                    except:
                        pass
            
            # Count successful extractions
            n_with_features = sum(1 for s in all_samples if s.get("audio_features"))
            print(f"  Successfully extracted: {n_with_features}")
    
    # Filter samples with features or valid labels
    valid_samples = [s for s in all_samples if s.get("audio_features") or s.get("quality", 0) > 0.5]
    print(f"\nValid samples: {len(valid_samples)}")
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(valid_samples)
    
    n = len(valid_samples)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_samples = valid_samples[:n_train]
    val_samples = valid_samples[n_train:n_train + n_val]
    test_samples = valid_samples[n_train + n_val:]
    
    # Count virus types
    virus_counts = {v.name: 0 for v in VirusType}
    for s in valid_samples:
        vt = VirusType(s.get("virus_type", 0))
        virus_counts[vt.name] += 1
    
    print(f"\nVirus type distribution:")
    for name, count in sorted(virus_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / len(valid_samples)
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Save splits
    print(f"\nSaving processed data...")
    
    with open(output_dir / "train.json", "w") as f:
        json.dump(train_samples, f)
    print(f"  Train: {len(train_samples)} samples")
    
    with open(output_dir / "val.json", "w") as f:
        json.dump(val_samples, f)
    print(f"  Val: {len(val_samples)} samples")
    
    with open(output_dir / "test.json", "w") as f:
        json.dump(test_samples, f)
    print(f"  Test: {len(test_samples)} samples")
    
    # Save metadata
    metadata = {
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "virus_types": [v.name for v in VirusType],
        "virus_counts": virus_counts,
        "datasets_used": list(set(s["dataset"] for s in valid_samples)),
        "feature_dim": 30,
    }
    
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)
    
    print(f"\n{'='*60}")
    print("Done! Data saved to:", output_dir)
    print(f"{'='*60}")
    print("\nNext step - train the model:")
    print("  python scripts/train.py --config configs/high_performance.yaml")


if __name__ == "__main__":
    main()

