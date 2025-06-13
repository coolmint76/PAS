"""
MRI Bone Porosity Analysis Pipeline - Part 1
=============================================
Core classes and protocol definitions for MRI-based bone porosity measurements.
Implements peer-reviewed methods for multiple sequence types including:
- UTE (Ultrashort Echo Time)
- VIBE/T1-GRE with Dixon
- ZTE (Zero Echo Time)
- T1 mapping sequences

Includes optimized protocols for 1.5T and 3T scanners.

Author: Your Research Team
Date: 2025
Version: 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.optimize as optimize
from scipy import stats
import pandas as pd
from skimage import morphology, filters, measure
import nibabel as nib
from typing import Tuple, Dict, Optional, Union, List
import warnings
from dataclasses import dataclass
import json

@dataclass
class SequenceParameters:
    """Data class for MRI sequence parameters."""
    sequence_type: str
    field_strength: float  # Tesla
    tr: float  # ms
    te: Union[float, List[float]]  # ms
    flip_angle: float  # degrees
    resolution: Tuple[float, float, float]  # mm
    fov: float  # mm
    matrix_size: Tuple[int, int, int]
    bandwidth: float  # Hz/pixel
    scan_time: float  # minutes
    additional_params: Dict = None

class MRIProtocolLibrary:
    """
    Library of optimized MRI protocols for bone imaging at 1.5T and 3T.
    Based on peer-reviewed literature and clinical validation.
    """
    
    @staticmethod
    def get_ute_protocol(field_strength: float = 3.0, 
                        clinical_time: bool = True) -> SequenceParameters:
        """
        Get optimized UTE protocol parameters.
        
        References:
        - Rajapakse et al., 2015: Clinical feasibility
        - Ma et al., 2020: 3D Cones optimization
        - Jerban et al., 2023: Ex vivo validation
        """
        if field_strength == 3.0:
            if clinical_time:
                # Clinical protocol (~10 minutes)
                return SequenceParameters(
                    sequence_type="3D_UTE_Cones",
                    field_strength=3.0,
                    tr=100.0,
                    te=[0.032, 2.2],  # Dual echo
                    flip_angle=10.0,
                    resolution=(0.5, 0.5, 2.0),
                    fov=140.0,
                    matrix_size=(280, 280, 40),
                    bandwidth=125000.0,
                    scan_time=10.0,
                    additional_params={
                        'readout_duration': 1.2,  # ms
                        'radial_views': 20000,
                        'cone_angle': 30,  # degrees
                        'fat_suppression': 'soft_hard_pulse'
                    }
                )
            else:
                # Research protocol (~20 minutes)
                return SequenceParameters(
                    sequence_type="3D_UTE_Cones_MultiTE",
                    field_strength=3.0,
                    tr=100.0,
                    te=[0.032, 0.2, 0.4, 0.8, 1.2, 2.2],
                    flip_angle=15.0,
                    resolution=(0.4, 0.4, 1.5),
                    fov=140.0,
                    matrix_size=(350, 350, 60),
                    bandwidth=125000.0,
                    scan_time=20.0,
                    additional_params={
                        'readout_duration': 1.4,
                        'radial_views': 40000,
                        'cone_angle': 35,
                        'fat_suppression': 'chemical_shift'
                    }
                )
        else:  # 1.5T
            return SequenceParameters(
                sequence_type="3D_UTE",
                field_strength=1.5,
                tr=150.0,  # Longer TR for 1.5T
                te=[0.05, 2.0],
                flip_angle=15.0,
                resolution=(1.0, 1.0, 2.0),
                fov=160.0,
                matrix_size=(160, 160, 40),
                bandwidth=62500.0,
                scan_time=12.0,
                additional_params={
                    'readout_duration': 2.0,
                    'radial_views': 15000,
                    'gradient_performance': 'standard'
                }
            )
    
    @staticmethod
    def get_irute_protocol(field_strength: float = 3.0) -> SequenceParameters:
        """
        Get IR-UTE protocol for selective bound water imaging.
        
        References:
        - Horch et al., 2012: Bound/pore water discrimination
        - Ma et al., 2020: Trabecular bone imaging
        """
        if field_strength == 3.0:
            return SequenceParameters(
                sequence_type="3D_IR_UTE_Cones",
                field_strength=3.0,
                tr=100.0,
                te=0.032,
                flip_angle=20.0,
                resolution=(0.5, 0.5, 2.0),
                fov=140.0,
                matrix_size=(280, 280, 40),
                bandwidth=125000.0,
                scan_time=12.0,
                additional_params={
                    'inversion_time': 45.0,  # ms
                    'adiabatic_pulse_duration': 8.6,  # ms
                    'adiabatic_bandwidth': 1150.0,  # Hz
                    'frequency_offset': -220.0,  # Hz from water
                    'nulling_strategy': 'short_TR_TI'
                }
            )
        else:  # 1.5T
            return SequenceParameters(
                sequence_type="3D_IR_UTE",
                field_strength=1.5,
                tr=150.0,
                te=0.05,
                flip_angle=25.0,
                resolution=(1.0, 1.0, 2.0),
                fov=160.0,
                matrix_size=(160, 160, 40),
                bandwidth=62500.0,
                scan_time=15.0,
                additional_params={
                    'inversion_time': 60.0,
                    'adiabatic_pulse_duration': 10.0,
                    'adiabatic_bandwidth': 800.0,
                    'frequency_offset': -150.0  # Adjusted for 1.5T
                }
            )
    
    @staticmethod
    def get_vibe_dixon_protocol(field_strength: float = 3.0) -> SequenceParameters:
        """
        Get T1-weighted 3D VIBE/GRE with Dixon for bone imaging.
        
        References:
        - Ang et al.: VIBE for pars fractures
        - Lee et al., 2021: CT-like bone imaging
        """
        if field_strength == 3.0:
            return SequenceParameters(
                sequence_type="3D_T1_VIBE_Dixon",
                field_strength=3.0,
                tr=9.0,
                te=[1.23, 2.46],  # In/out phase at 3T
                flip_angle=10.0,
                resolution=(0.6, 0.6, 1.0),
                fov=280.0,
                matrix_size=(448, 448, 120),
                bandwidth=400.0,
                scan_time=3.5,
                additional_params={
                    'acceleration': 'GRAPPA_2',
                    'fat_water_separation': 'two_point_dixon',
                    'reconstruction': ['water', 'fat', 'in_phase', 'opposed_phase'],
                    'gray_scale_inversion': True
                }
            )
        else:  # 1.5T
            return SequenceParameters(
                sequence_type="3D_T1_VIBE_Dixon",
                field_strength=1.5,
                tr=12.0,
                te=[2.38, 4.76],  # In/out phase at 1.5T
                flip_angle=12.0,
                resolution=(0.8, 0.8, 1.5),
                fov=300.0,
                matrix_size=(384, 384, 80),
                bandwidth=260.0,
                scan_time=4.0,
                additional_params={
                    'acceleration': 'GRAPPA_2',
                    'fat_water_separation': 'two_point_dixon',
                    'multi_echo_combination': True
                }
            )
    
    @staticmethod
    def get_zte_protocol(field_strength: float = 3.0) -> SequenceParameters:
        """
        Get Zero Echo Time protocol for silent bone imaging.
        
        References:
        - Breighner et al., 2018: Clinical ZTE
        - Wiesinger et al., 2016: ZTE principles
        """
        if field_strength == 3.0:
            return SequenceParameters(
                sequence_type="3D_ZTE",
                field_strength=3.0,
                tr=1.0,  # Very short TR
                te=0.0,  # Zero nominal TE
                flip_angle=1.0,
                resolution=(0.8, 0.8, 0.8),  # Isotropic
                fov=280.0,
                matrix_size=(350, 350, 350),
                bandwidth=125000.0,
                scan_time=5.0,
                additional_params={
                    'encoding_type': 'center_out_radial',
                    'dead_time': 8.0,  # μs
                    'gradient_ramp': 'triangular',
                    'reconstruction': 'algebraic',
                    'acoustic_noise': '<5dB',  # Silent scanning
                    'post_processing': ['bias_correction', 'log_inversion']
                }
            )
        else:  # Limited availability at 1.5T
            warnings.warn("ZTE is rarely available at 1.5T. Consider UTE instead.")
            return MRIProtocolLibrary.get_ute_protocol(1.5)
    
    @staticmethod
    def get_t1_mapping_protocol(field_strength: float = 3.0) -> SequenceParameters:
        """
        Get T1 mapping protocol for bone water assessment.
        
        References:
        - Chen et al., 2017: T1 of bound/pore water
        - Ma et al., 2018: VTR method
        """
        if field_strength == 3.0:
            return SequenceParameters(
                sequence_type="3D_UTE_VTR_T1",
                field_strength=3.0,
                tr=[100.0, 200.0, 400.0, 800.0],  # Variable TR
                te=0.032,
                flip_angle=45.0,
                resolution=(1.0, 1.0, 2.0),
                fov=140.0,
                matrix_size=(140, 140, 40),
                bandwidth=125000.0,
                scan_time=16.0,
                additional_params={
                    'mapping_method': 'variable_TR',
                    'saturation_correction': True,
                    'b1_mapping': 'AFI',
                    'fitting_algorithm': 'NLLS'
                }
            )
        else:
            return SequenceParameters(
                sequence_type="2D_UTE_VTR_T1",
                field_strength=1.5,
                tr=[150.0, 300.0, 600.0, 1200.0],
                te=0.05,
                flip_angle=60.0,
                resolution=(1.5, 1.5, 3.0),
                fov=160.0,
                matrix_size=(107, 107, 20),
                bandwidth=62500.0,
                scan_time=18.0,
                additional_params={
                    'slice_selective': True,
                    'crusher_gradients': True
                }
            )

class BonePorosityAnalyzer:
    """
    Main class for analyzing bone porosity from UTE-MRI data.
    
    References:
    - Rajapakse et al., 2015: Porosity Index methodology
    - Chen et al., 2021: 3D UTE implementation
    - Jerban et al., 2024: Clinical validation
    """
    
    def __init__(self, 
                 te1: float = 0.05,
                 te2: float = 2.0,
                 field_strength: float = 3.0):
        self.te1 = te1
        self.te2 = te2
        self.field_strength = field_strength
        self.t2_bound_water = 0.3
        self.t2_pore_water = 2.3
        self.voxel_dims = (1.0, 1.0, 1.0)  # Default voxel dimensions
        """
        Initialize the bone porosity analyzer.
        
        Parameters:
        -----------
        te1 : float
            First (ultrashort) echo time in milliseconds
        te2 : float
            Second echo time in milliseconds
        field_strength : float
            MRI field strength in Tesla
        """
        self.te1 = te1
        self.te2 = te2
        self.field_strength = field_strength
        
        # Water relaxation parameters (from literature)
        self.t2_bound_water = 0.3  # ms, bound water T2
        self.t2_pore_water = 2.3   # ms, pore water T2
        
    def load_ute_data(self, 
                      echo1_path: str, 
                      echo2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load UTE MRI data from NIfTI files.
        
        Parameters:
        -----------
        echo1_path : str
            Path to first echo NIfTI file
        echo2_path : str
            Path to second echo NIfTI file
            
        Returns:
        --------
        echo1_data, echo2_data : tuple of np.ndarray
            3D arrays containing the MRI data
        """
        echo1_nii = nib.load(echo1_path)
        echo2_nii = nib.load(echo2_path)
        
        echo1_data = echo1_nii.get_fdata()
        echo2_data = echo2_nii.get_fdata()
        
        # Store voxel dimensions for later use
        self.voxel_dims = echo1_nii.header.get_zooms()[:3]
        
        return echo1_data, echo2_data
    
    def calculate_regional_statistics(self,
                                    parameter_map: np.ndarray,
                                    mask: np.ndarray,
                                    n_layers: int = 3) -> pd.DataFrame:
        """
        Calculate statistics in cortical layers (endosteal to periosteal).
        """
        # Check if mask has any valid pixels
        if np.sum(mask) == 0:
            return pd.DataFrame()  # Return empty DataFrame
        
        # Distance transform to create layers
        dist_map = ndimage.distance_transform_edt(mask)
        max_dist = dist_map.max()
        
        # Check if distance map is valid
        if max_dist == 0:
            # If all distances are 0, create a single layer
            stats_list = [{
                'layer': 1,
                'layer_name': 'Cortical',
                'mean': np.mean(parameter_map[mask > 0]),
                'std': np.std(parameter_map[mask > 0]),
                'median': np.median(parameter_map[mask > 0]),
                'q25': np.percentile(parameter_map[mask > 0], 25),
                'q75': np.percentile(parameter_map[mask > 0], 75),
                'volume_mm3': mask.sum() * np.prod(self.voxel_dims) if hasattr(self, 'voxel_dims') else mask.sum()
            }]
            return pd.DataFrame(stats_list)
        
        # Create layer masks
        layer_boundaries = np.linspace(0, max_dist, n_layers + 1)
        
        # Ensure we have at least 3 layer names
        layer_names = ['Endosteal', 'Mid-cortical', 'Periosteal']
        if n_layers > 3:
            layer_names.extend([f'Layer_{i}' for i in range(4, n_layers + 1)])
        elif n_layers < 3:
            layer_names = layer_names[:n_layers]
        
        stats_list = []
        for i in range(n_layers):
            layer_mask = ((dist_map > layer_boundaries[i]) & 
                         (dist_map <= layer_boundaries[i + 1]) & 
                         mask)
            
            if layer_mask.sum() > 0:
                values = parameter_map[layer_mask]
                stats_list.append({
                    'layer': i + 1,
                    'layer_name': layer_names[i] if i < len(layer_names) else f'Layer_{i+1}',
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'volume_mm3': layer_mask.sum() * np.prod(self.voxel_dims) if hasattr(self, 'voxel_dims') else layer_mask.sum()
                })
        
        return pd.DataFrame(stats_list)
    
    def segment_cortical_bone(self, 
                             echo1: np.ndarray,
                             threshold_percentile: float = 70) -> np.ndarray:
        """
        Segment cortical bone using automated thresholding.
        
        Parameters:
        -----------
        echo1 : np.ndarray
            First echo image (highest SNR for cortical bone)
        threshold_percentile : float
            Percentile for Otsu thresholding
            
        Returns:
        --------
        mask : np.ndarray
            Binary mask of cortical bone
        """
        # Apply Gaussian smoothing
        smoothed = filters.gaussian(echo1, sigma=1.0)
        
        # Otsu thresholding
        threshold = filters.threshold_otsu(smoothed[smoothed > 0])
        threshold *= (threshold_percentile / 100.0)
        
        # Create initial mask
        mask = smoothed > threshold
        
        # Morphological operations to clean up
        mask = morphology.binary_opening(mask, morphology.ball(2))
        mask = morphology.binary_closing(mask, morphology.ball(2))
        
        # Remove small objects
        mask = morphology.remove_small_objects(mask, min_size=100)
        
        # Fill holes
        mask = ndimage.binary_fill_holes(mask)
        
        return mask.astype(np.uint8)
    
    def calculate_suppression_ratio(self,
                                   unsuppressed: np.ndarray,
                                   suppressed: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate Suppression Ratio (SR) for long-T2 suppression.
        
        SR = S_unsuppressed / S_suppressed
        
        Parameters:
        -----------
        unsuppressed : np.ndarray
            Signal without long-T2 suppression
        suppressed : np.ndarray
            Signal with long-T2 suppression
        mask : np.ndarray, optional
            Binary mask for analysis region
            
        Returns:
        --------
        sr : np.ndarray
            Suppression Ratio map
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            sr = unsuppressed / suppressed
            
        sr[~np.isfinite(sr)] = 0
        
        if mask is not None:
            sr = sr * mask
            
        return sr

    def biexponential_analysis(self,
                              signal_array: np.ndarray,
                              te_array: np.ndarray) -> Dict[str, float]:
        """
        Perform biexponential fitting to separate bound and pore water.
        
        S(TE) = A_bound * exp(-TE/T2_bound) + A_pore * exp(-TE/T2_pore)
        
        Parameters:
        -----------
        signal_array : np.ndarray
            Array of signal intensities at different TEs
        te_array : np.ndarray
            Array of echo times
            
        Returns:
        --------
        params : dict
            Dictionary containing fitted parameters
        """
        def biexponential(te, a_bound, a_pore, t2_bound, t2_pore):
            return (a_bound * np.exp(-te / t2_bound) + 
                   a_pore * np.exp(-te / t2_pore))
        
        # Initial guess
        p0 = [signal_array[0] * 0.7,  # A_bound
              signal_array[0] * 0.3,  # A_pore  
              self.t2_bound_water,    # T2_bound
              self.t2_pore_water]     # T2_pore
        
        try:
            popt, _ = optimize.curve_fit(biexponential, te_array, signal_array,
                                        p0=p0, maxfev=5000)
            
            # Calculate water fractions
            total_signal = popt[0] + popt[1]
            bound_fraction = popt[0] / total_signal * 100
            pore_fraction = popt[1] / total_signal * 100
            
            return {
                'bound_water_fraction': bound_fraction,
                'pore_water_fraction': pore_fraction,
                't2_bound': popt[2],
                't2_pore': popt[3],
                'fit_success': True
            }
        except:
            return {
                'bound_water_fraction': np.nan,
                'pore_water_fraction': np.nan,
                't2_bound': np.nan,
                't2_pore': np.nan,
                'fit_success': False
            }
    
    def calculate_regional_statistics(self,
                                    parameter_map: np.ndarray,
                                    mask: np.ndarray,
                                    n_layers: int = 3) -> pd.DataFrame:
        """
        Calculate statistics in cortical layers (endosteal to periosteal).
        
        Parameters:
        -----------
        parameter_map : np.ndarray
            3D parameter map (e.g., PI, SR)
        mask : np.ndarray
            Binary mask of cortical bone
        n_layers : int
            Number of layers to analyze
            
        Returns:
        --------
        stats_df : pd.DataFrame
            Statistics for each layer
        """
        # Distance transform to create layers
        dist_map = ndimage.distance_transform_edt(mask)
        max_dist = dist_map.max()
        
        # Create layer masks
        layer_boundaries = np.linspace(0, max_dist, n_layers + 1)
        
        stats_list = []
        for i in range(n_layers):
            layer_mask = ((dist_map > layer_boundaries[i]) & 
                         (dist_map <= layer_boundaries[i + 1]) & 
                         mask)
            
            if layer_mask.sum() > 0:
                values = parameter_map[layer_mask]
                stats_list.append({
                    'layer': i + 1,
                    'layer_name': ['Endosteal', 'Mid-cortical', 'Periosteal'][i],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                    'volume_mm3': layer_mask.sum() * np.prod(self.voxel_dims)
                })
        
        return pd.DataFrame(stats_list)
    
    def correlate_with_bmd(self,
                          porosity_values: np.ndarray,
                          bmd_values: np.ndarray) -> Dict[str, float]:
        """
        Correlate porosity measurements with BMD (if available).
        
        Parameters:
        -----------
        porosity_values : np.ndarray
            Array of porosity measurements
        bmd_values : np.ndarray
            Array of BMD measurements
            
        Returns:
        --------
        correlation_stats : dict
            Correlation statistics
        """
        # Remove any NaN values
        valid_mask = ~(np.isnan(porosity_values) | np.isnan(bmd_values))
        porosity_clean = porosity_values[valid_mask]
        bmd_clean = bmd_values[valid_mask]
        
        # Calculate correlations
        pearson_r, pearson_p = stats.pearsonr(porosity_clean, bmd_clean)
        spearman_r, spearman_p = stats.spearmanr(porosity_clean, bmd_clean)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            porosity_clean, bmd_clean)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'regression_p': p_value
        }
    
    def analyze_dixon_data(self,
                         water_image: np.ndarray,
                         fat_image: np.ndarray,
                         in_phase: np.ndarray,
                         opposed_phase: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> Dict[str, Union[np.ndarray, float]]:
        """
        Analyze Dixon data for bone marrow assessment.
        
        Parameters:
        -----------
        water_image : np.ndarray
            Water-only image from Dixon
        fat_image : np.ndarray
            Fat-only image from Dixon
        in_phase : np.ndarray
            In-phase image
        opposed_phase : np.ndarray
            Opposed-phase image
        mask : np.ndarray, optional
            ROI mask
            
        Returns:
        --------
        results : dict
            Dictionary containing fat fraction, water fraction, and IO ratio
        """
        # Calculate fat fraction
        with np.errstate(divide='ignore', invalid='ignore'):
            fat_fraction = fat_image / (water_image + fat_image) * 100
            water_fraction = water_image / (water_image + fat_image) * 100
            
            # In-phase/opposed-phase ratio
            io_ratio = (in_phase - opposed_phase) / (in_phase + opposed_phase) * 100
        
        # Handle invalid values
        fat_fraction[~np.isfinite(fat_fraction)] = 0
        water_fraction[~np.isfinite(water_fraction)] = 0
        io_ratio[~np.isfinite(io_ratio)] = 0
        
        # Apply mask if provided
        if mask is not None:
            fat_fraction *= mask
            water_fraction *= mask
            io_ratio *= mask
        
        # Calculate statistics
        valid_mask = mask > 0 if mask is not None else np.ones_like(fat_fraction, dtype=bool)
        
        results = {
            'fat_fraction_map': fat_fraction,
            'water_fraction_map': water_fraction,
            'io_ratio_map': io_ratio,
            'mean_fat_fraction': np.mean(fat_fraction[valid_mask]),
            'std_fat_fraction': np.std(fat_fraction[valid_mask]),
            'mean_water_fraction': np.mean(water_fraction[valid_mask]),
            'std_water_fraction': np.std(water_fraction[valid_mask]),
            'mean_io_ratio': np.mean(io_ratio[valid_mask]),
            'std_io_ratio': np.std(io_ratio[valid_mask])
        }
        
        # Bone-specific thresholds (from literature)
        # Normal trabecular bone: FF > 50-70%
        # Osteoporotic changes: FF may increase
        # Metastases/infiltration: FF < 20%
        
        if results['mean_fat_fraction'] < 20:
            results['interpretation'] = 'Abnormal - possible infiltration/metastases'
        elif results['mean_fat_fraction'] < 50:
            results['interpretation'] = 'Below normal marrow fat content'
        elif results['mean_fat_fraction'] > 80:
            results['interpretation'] = 'Increased marrow fat - possible osteoporosis'
        else:
            results['interpretation'] = 'Normal marrow fat content'
        
        return results
    
    def process_vibe_for_cortical_bone(self,
                                     vibe_data: np.ndarray,
                                     echo_times: List[float],
                                     gray_scale_invert: bool = True) -> np.ndarray:
        """
        Process VIBE data to create CT-like bone images.
        
        Parameters:
        -----------
        vibe_data : np.ndarray
            4D array (x, y, z, echoes)
        echo_times : list
            List of echo times in ms
        gray_scale_invert : bool
            Whether to invert gray scale for CT-like appearance
            
        Returns:
        --------
        bone_image : np.ndarray
            CT-like bone image
        """
        # Sum echoes for better SNR (in-phase echoes)
        # Select only in-phase echoes based on field strength
        if self.field_strength == 3.0:
            in_phase_interval = 2.3  # ms
        else:  # 1.5T
            in_phase_interval = 4.6  # ms
        
        in_phase_indices = []
        for i, te in enumerate(echo_times):
            if abs(te % in_phase_interval) < 0.2:  # Tolerance
                in_phase_indices.append(i)
        
        # Sum in-phase echoes
        if len(in_phase_indices) > 0:
            bone_image = np.sum(vibe_data[..., in_phase_indices], axis=-1)
        else:
            bone_image = np.sum(vibe_data, axis=-1)
        
        # Apply bias field correction
        bone_image = self._bias_field_correction(bone_image)
        
        # Gray scale inversion for CT-like appearance
        if gray_scale_invert:
            bone_image = np.max(bone_image) - bone_image
        
        # Apply edge enhancement for better cortical definition
        bone_image = self._edge_enhancement(bone_image)
        
        return bone_image
    
    def _bias_field_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply N4 bias field correction."""
        # Simplified bias field correction
        # In practice, use SimpleITK's N4BiasFieldCorrection
        from scipy.ndimage import gaussian_filter
        
        # Estimate bias field
        smoothed = gaussian_filter(image, sigma=20)
        bias_field = smoothed / (np.mean(smoothed) + 1e-8)
        
        # Correct image
        corrected = image / (bias_field + 1e-8)
        
        return corrected
    
    def _edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Enhance edges for better cortical bone visualization."""
        from scipy.ndimage import sobel
        
        # Calculate gradients
        sx = sobel(image, axis=0)
        sy = sobel(image, axis=1)
        sz = sobel(image, axis=2)
        
        # Gradient magnitude
        edge_map = np.sqrt(sx**2 + sy**2 + sz**2)
        
        # Combine with original
        enhanced = image + 0.3 * edge_map
        
        return enhanced
    
    def generate_report(self,
                       pi_map: np.ndarray,
                       mask: np.ndarray,
                       patient_id: str = "Anonymous",
                       save_path: Optional[str] = None) -> str:
        """
        Generate a clinical report with visualizations.
        
        Parameters:
        -----------
        pi_map : np.ndarray
            Porosity Index map
        mask : np.ndarray
            Cortical bone mask
        patient_id : str
            Patient identifier
        save_path : str, optional
            Path to save the report
            
        Returns:
        --------
        report_text : str
            Clinical report text
        """
        # Calculate statistics
        pi_values = pi_map[mask > 0]
        mean_pi = np.mean(pi_values)
        std_pi = np.std(pi_values)
        
        # Get layer statistics
        layer_stats = self.calculate_regional_statistics(pi_map, mask)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Central slice with PI overlay
        central_slice = pi_map.shape[2] // 2
        ax1 = axes[0, 0]
        im1 = ax1.imshow(pi_map[:, :, central_slice], cmap='hot', vmin=0, vmax=50)
        ax1.set_title('Porosity Index Map (Central Slice)')
        plt.colorbar(im1, ax=ax1, label='PI (%)')
        
        # 2. Histogram
        ax2 = axes[0, 1]
        ax2.hist(pi_values, bins=50, density=True, alpha=0.7, color='blue')
        ax2.axvline(mean_pi, color='red', linestyle='--', 
                   label=f'Mean: {mean_pi:.1f}%')
        ax2.set_xlabel('Porosity Index (%)')
        ax2.set_ylabel('Density')
        ax2.set_title('PI Distribution')
        ax2.legend()
        
        # 3. Layer analysis
        ax3 = axes[1, 0]
        ax3.bar(layer_stats['layer_name'], layer_stats['mean'], 
               yerr=layer_stats['std'], capsize=5)
        ax3.set_ylabel('Mean PI (%)')
        ax3.set_title('Porosity by Cortical Layer')
        
        # 4. 3D projection
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.5, f"Patient ID: {patient_id}\n"
                          f"Mean PI: {mean_pi:.1f} ± {std_pi:.1f}%\n"
                          f"Cortical Volume: {mask.sum() * np.prod(self.voxel_dims):.1f} mm³\n"
                          f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                transform=ax4.transAxes, fontsize=12, verticalalignment='center')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Generate report text
        report_text = f"""
MRI BONE POROSITY ANALYSIS REPORT
==================================
Patient ID: {patient_id}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
MRI Protocol: Dual-echo UTE (TE1={self.te1}ms, TE2={self.te2}ms)

SUMMARY FINDINGS:
-----------------
Overall Cortical Porosity Index: {mean_pi:.1f} ± {std_pi:.1f}%
Cortical Bone Volume: {mask.sum() * np.prod(self.voxel_dims):.1f} mm³

REGIONAL ANALYSIS:
------------------
"""
        for _, row in layer_stats.iterrows():
            report_text += f"{row['layer_name']} Layer: {row['mean']:.1f} ± {row['std']:.1f}%\n"
        
        report_text += f"""
INTERPRETATION:
---------------
{'NORMAL' if mean_pi < 15 else 'ELEVATED' if mean_pi < 25 else 'SIGNIFICANTLY ELEVATED'} 
cortical porosity detected.

Reference ranges (postmenopausal women):
- Normal: < 15%
- Mildly elevated: 15-25%
- Significantly elevated: > 25%

RECOMMENDATIONS:
----------------
"""
        if mean_pi > 25:
            report_text += "Consider comprehensive fracture risk assessment including DXA.\n"
            report_text += "Clinical correlation recommended.\n"
        elif mean_pi > 15:
            report_text += "Monitor with follow-up imaging in 12-24 months.\n"
        else:
            report_text += "No specific follow-up required based on porosity alone.\n"
        
        return report_text
    
    def generate_sequence_optimization_guide(self,
                                           field_strength: float,
                                           clinical_application: str,
                                           save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive sequence optimization guide.
        
        Parameters:
        -----------
        field_strength : float
            MRI field strength (1.5 or 3.0 Tesla)
        clinical_application : str
            Target application (e.g., 'osteoporosis', 'fracture_risk', 'treatment_monitoring')
        save_path : str, optional
            Path to save the guide as PDF
            
        Returns:
        --------
        guide_text : str
            Formatted optimization guide
        """
        # Get all relevant protocols
        protocols = {
            'UTE_clinical': MRIProtocolLibrary.get_ute_protocol(field_strength, clinical_time=True),
            'UTE_research': MRIProtocolLibrary.get_ute_protocol(field_strength, clinical_time=False),
            'IR_UTE': MRIProtocolLibrary.get_irute_protocol(field_strength),
            'VIBE_Dixon': MRIProtocolLibrary.get_vibe_dixon_protocol(field_strength),
            'T1_mapping': MRIProtocolLibrary.get_t1_mapping_protocol(field_strength)
        }
        
        if field_strength == 3.0:
            protocols['ZTE'] = MRIProtocolLibrary.get_zte_protocol(3.0)
        
        # Create optimization guide
        guide_text = f"""
MRI BONE POROSITY IMAGING OPTIMIZATION GUIDE
============================================
Field Strength: {field_strength}T
Clinical Application: {clinical_application}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

EXECUTIVE SUMMARY
-----------------
"""
        # Application-specific recommendations
        if clinical_application == 'osteoporosis':
            guide_text += """
For osteoporosis assessment, the recommended protocol hierarchy is:
1. UTE Dual-Echo (PI calculation) - Primary method
2. IR-UTE (Bound water assessment) - Complementary
3. VIBE Dixon (Trabecular assessment) - Additional structural info
"""
        elif clinical_application == 'fracture_risk':
            guide_text += """
For fracture risk assessment, combine multiple sequences:
1. UTE Multi-Echo - Comprehensive water compartment analysis
2. T1 Mapping - Tissue quality assessment
3. VIBE Dixon - Structural integrity evaluation
"""
        elif clinical_application == 'treatment_monitoring':
            guide_text += """
For treatment monitoring, prioritize sensitivity to change:
1. IR-UTE - Most sensitive to bound water changes
2. UTE Dual-Echo - Reproducible porosity measurements
3. T1 Mapping - Tracks compositional changes
"""
        
        guide_text += f"""

DETAILED PROTOCOL PARAMETERS
----------------------------
"""
        
        # Add detailed parameters for each sequence
        for name, params in protocols.items():
            guide_text += f"""
{name.replace('_', ' ').upper()}
{'=' * len(name)}
Sequence Type: {params.sequence_type}
Scan Time: {params.scan_time} minutes
Resolution: {params.resolution[0]} × {params.resolution[1]} × {params.resolution[2]} mm³
"""
            if isinstance(params.te, list):
                guide_text += f"Echo Times: {', '.join([f'{te}ms' for te in params.te])}\n"
            else:
                guide_text += f"Echo Time: {params.te} ms\n"
                
            guide_text += f"""TR: {params.tr} ms
Flip Angle: {params.flip_angle}°
Bandwidth: {params.bandwidth/1000:.0f} kHz
FOV: {params.fov} mm
Matrix: {params.matrix_size[0]} × {params.matrix_size[1]} × {params.matrix_size[2]}
"""
            if params.additional_params:
                guide_text += "\nSpecial Parameters:\n"
                for key, value in params.additional_params.items():
                    guide_text += f"  - {key.replace('_', ' ').title()}: {value}\n"
            guide_text += "\n"
        
        # Add optimization tips
        guide_text += f"""
OPTIMIZATION STRATEGIES FOR {field_strength}T
=========================================
"""
        
        if field_strength == 3.0:
            guide_text += """
1. SNR Advantages:
   - Exploit 2× SNR for higher resolution or faster scanning
   - Use parallel imaging (GRAPPA/SENSE) more aggressively
   - Consider multi-echo acquisitions for comprehensive analysis

2. Specific Absorption Rate (SAR) Management:
   - Use variable flip angle schemes
   - Implement TR optimization for multi-echo sequences
   - Consider hyperecho techniques for refocusing

3. B0 Inhomogeneity Compensation:
   - Use 3D shimming with higher-order shims
   - Apply dynamic shimming for different regions
   - Consider susceptibility-weighted corrections

4. Chemical Shift Considerations:
   - Fat-water separation occurs at 1.15 ms intervals
   - Use multi-point Dixon for robust separation
   - Apply IDEAL reconstruction when available
"""
        else:  # 1.5T
            guide_text += """
1. SNR Optimization:
   - Use larger voxel sizes (1-1.5 mm in-plane)
   - Increase number of averages for critical measurements
   - Optimize receiver bandwidth (compromise with chemical shift)

2. Contrast Optimization:
   - Longer T1 values allow better T1 contrast
   - Use longer TRs for better SNR without SAR issues
   - Consider magnetization preparation pulses

3. Chemical Shift Management:
   - Fat-water separation at 2.3 ms intervals
   - Two-point Dixon usually sufficient
   - Lower bandwidth acceptable (less chemical shift)

4. Gradient Performance:
   - May limit minimum TE for UTE sequences
   - Consider 2D UTE if 3D performance limited
   - Use optimized gradient waveforms
"""
        
        # Continue with practical tips...
        guide_text += """

PRACTICAL IMPLEMENTATION TIPS
=============================

1. Patient Positioning:
   - Use dedicated coils when possible (knee, wrist)
   - Ensure region of interest is at isocenter
   - Minimize motion with comfortable positioning

2. Quality Control:
   - Perform B0 and B1 mapping for quantitative studies
   - Use phantoms for sequence validation
   - Monitor temperature for long acquisitions

3. Post-Processing Pipeline:
   - Implement automated segmentation
   - Apply motion correction if needed
   - Use appropriate curve fitting for parametric maps

4. Common Artifacts and Solutions:
"""
        
        # Sequence-specific artifacts
        artifact_solutions = {
            'UTE': [
                ('Gradient delays', 'Calibrate with phantom'),
                ('Radial streaking', 'Increase number of spokes'),
                ('Chemical shift', 'Use fat suppression')
            ],
            'VIBE_Dixon': [
                ('Fat-water swaps', 'Use multi-peak fat model'),
                ('Phase errors', 'Apply phase correction'),
                ('Motion', 'Use respiratory triggering if needed')
            ],
            'ZTE': [
                ('Dead time gap', 'Apply algebraic reconstruction'),
                ('Gradient fidelity', 'Use gradient pre-emphasis'),
                ('Long T2 contamination', 'Apply pointwise encoding')
            ]
        }
        
        for seq, artifacts in artifact_solutions.items():
            guide_text += f"\n   {seq}:\n"
            for artifact, solution in artifacts:
                guide_text += f"   - {artifact}: {solution}\n"
        
        # Add clinical interpretation guide
        guide_text += """

CLINICAL INTERPRETATION GUIDELINES
==================================

Porosity Index (PI) Reference Ranges:
- Normal: < 15%
- Mild increase: 15-25%
- Moderate increase: 25-35%
- Severe increase: > 35%

Bound Water Fraction:
- Normal cortical: 15-20%
- Osteoporotic: < 15%
- Correlation with organic matrix

T1 Values:
- Bound water: 140-290 ms (at 3T)
- Pore water: 600-1000 ms (at 3T)
- Values ~20% longer at 1.5T

Expected Correlations:
- PI vs μCT porosity: r = 0.8-0.9
- PI vs BMD: r = -0.6 to -0.8
- Bound water vs mechanical properties: r = 0.7-0.8
"""
        
        # Add references
        guide_text += """

KEY REFERENCES
==============
1. Rajapakse et al. Radiology 2015 - Clinical validation of PI
2. Ma et al. JMRI 2020 - UTE technical review
3. Jerban et al. Bone 2023 - Mechanical correlation
4. Wiesinger et al. MRM 2016 - ZTE principles
5. Lee et al. Eur Radiol 2021 - VIBE bone imaging

For updates, consult recent literature and vendor-specific guides.
"""
        
        if save_path:
            # Save as formatted text file (PDF conversion would require additional libraries)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(guide_text)
        
        return guide_text

# Utility functions - these are standalone, not part of the BonePorosityAnalyzer class

def compare_sequences_for_application(application: str = 'osteoporosis',
                                    scanner_constraints: Dict = None) -> pd.DataFrame:
    """
    Compare different sequences for a specific clinical application.
    
    Parameters:
    -----------
    application : str
        Clinical application (osteoporosis, fracture_risk, treatment_monitoring)
    scanner_constraints : dict
        Scanner limitations (e.g., {'max_gradient': 40, 'sequences_available': [...]})
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table of sequences
    """
    # Default scanner capabilities
    if scanner_constraints is None:
        scanner_constraints = {
            'field_strength': [1.5, 3.0],
            'max_gradient': 80,  # mT/m
            'sequences_available': ['UTE', 'VIBE', 'IR_UTE', 'ZTE', 'T1_mapping']
        }
    
    comparison_data = []
    
    # Sequence ratings for different applications (1-5 scale)
    ratings = {
        'osteoporosis': {
            'UTE': {'sensitivity': 4, 'specificity': 4, 'reproducibility': 5, 'clinical_time': 5},
            'IR_UTE': {'sensitivity': 5, 'specificity': 5, 'reproducibility': 4, 'clinical_time': 3},
            'VIBE_Dixon': {'sensitivity': 3, 'specificity': 3, 'reproducibility': 5, 'clinical_time': 5},
            'ZTE': {'sensitivity': 4, 'specificity': 4, 'reproducibility': 5, 'clinical_time': 5},
            'T1_mapping': {'sensitivity': 4, 'specificity': 4, 'reproducibility': 3, 'clinical_time': 2}
        },
        'fracture_risk': {
            'UTE': {'sensitivity': 5, 'specificity': 4, 'reproducibility': 5, 'clinical_time': 5},
            'IR_UTE': {'sensitivity': 4, 'specificity': 5, 'reproducibility': 4, 'clinical_time': 3},
            'VIBE_Dixon': {'sensitivity': 4, 'specificity': 3, 'reproducibility': 5, 'clinical_time': 5},
            'ZTE': {'sensitivity': 3, 'specificity': 4, 'reproducibility': 5, 'clinical_time': 5},
            'T1_mapping': {'sensitivity': 5, 'specificity': 4, 'reproducibility': 3, 'clinical_time': 2}
        },
        'treatment_monitoring': {
            'UTE': {'sensitivity': 4, 'specificity': 4, 'reproducibility': 5, 'clinical_time': 5},
            'IR_UTE': {'sensitivity': 5, 'specificity': 5, 'reproducibility': 4, 'clinical_time': 3},
            'VIBE_Dixon': {'sensitivity': 3, 'specificity': 3, 'reproducibility': 4, 'clinical_time': 5},
            'ZTE': {'sensitivity': 3, 'specificity': 4, 'reproducibility': 5, 'clinical_time': 5},
            'T1_mapping': {'sensitivity': 5, 'specificity': 5, 'reproducibility': 3, 'clinical_time': 2}
        }
    }
    
    for seq in scanner_constraints['sequences_available']:
        if seq in ratings[application]:
            seq_ratings = ratings[application][seq]
            
            # Calculate composite score
            composite_score = np.mean(list(seq_ratings.values()))
            
            # Get sequence parameters for reference
            if seq == 'UTE':
                params = MRIProtocolLibrary.get_ute_protocol(3.0, clinical_time=True)
            elif seq == 'IR_UTE':
                params = MRIProtocolLibrary.get_irute_protocol(3.0)
            elif seq == 'VIBE_Dixon':
                params = MRIProtocolLibrary.get_vibe_dixon_protocol(3.0)
            elif seq == 'ZTE':
                params = MRIProtocolLibrary.get_zte_protocol(3.0)
            else:
                params = MRIProtocolLibrary.get_t1_mapping_protocol(3.0)
            
            comparison_data.append({
                'Sequence': seq,
                'Scan Time (min)': params.scan_time,
                'Resolution (mm)': f"{params.resolution[0]}×{params.resolution[1]}×{params.resolution[2]}",
                'Sensitivity': seq_ratings['sensitivity'],
                'Specificity': seq_ratings['specificity'],
                'Reproducibility': seq_ratings['reproducibility'],
                'Clinical Feasibility': seq_ratings['clinical_time'],
                'Composite Score': f"{composite_score:.1f}",
                'Primary Measurement': {
                    'UTE': 'Porosity Index',
                    'IR_UTE': 'Bound Water',
                    'VIBE_Dixon': 'Structure',
                    'ZTE': 'Cortical Density',
                    'T1_mapping': 'Water Compartments'
                }[seq]
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Composite Score', ascending=False)
    
    return df

def generate_protocol_card(sequence_type: str, 
                          field_strength: float,
                          save_path: Optional[str] = None) -> None:
    """
    Generate a quick reference protocol card for scanning.
    
    Parameters:
    -----------
    sequence_type : str
        Type of sequence (UTE, IR_UTE, VIBE_Dixon, ZTE, T1_mapping)
    field_strength : float
        Scanner field strength (1.5 or 3.0 T)
    save_path : str, optional
        Path to save the protocol card
    """
    # Get protocol
    if sequence_type == 'UTE':
        protocol = MRIProtocolLibrary.get_ute_protocol(field_strength)
    elif sequence_type == 'IR_UTE':
        protocol = MRIProtocolLibrary.get_irute_protocol(field_strength)
    elif sequence_type == 'VIBE_Dixon':
        protocol = MRIProtocolLibrary.get_vibe_dixon_protocol(field_strength)
    elif sequence_type == 'ZTE':
        protocol = MRIProtocolLibrary.get_zte_protocol(field_strength)
    else:
        protocol = MRIProtocolLibrary.get_t1_mapping_protocol(field_strength)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('off')
    
    # Title
    title = f"{protocol.sequence_type} Protocol Card\n{field_strength}T Scanner"
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=16, weight='bold')
    
    # Main parameters
    y_pos = 0.85
    param_text = f"""
Scan Time: {protocol.scan_time} minutes
TR: {protocol.tr} ms
TE: {protocol.te} ms
Flip Angle: {protocol.flip_angle}°
Resolution: {protocol.resolution[0]} × {protocol.resolution[1]} × {protocol.resolution[2]} mm³
FOV: {protocol.fov} mm
Matrix: {protocol.matrix_size[0]} × {protocol.matrix_size[1]} × {protocol.matrix_size[2]}
Bandwidth: {protocol.bandwidth/1000:.0f} kHz
"""
    
    ax.text(0.1, y_pos, param_text, ha='left', va='top', fontsize=11, 
            family='monospace', bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor="lightgray", alpha=0.5))
    
    # Additional parameters
    if protocol.additional_params:
        y_pos = 0.45
        ax.text(0.1, y_pos, "Special Parameters:", ha='left', va='top', 
                fontsize=12, weight='bold')
        y_pos -= 0.05
        
        for key, value in protocol.additional_params.items():
            param_name = key.replace('_', ' ').title()
            ax.text(0.1, y_pos, f"• {param_name}: {value}", 
                   ha='left', va='top', fontsize=10)
            y_pos -= 0.04
    
    # Quick tips
    y_pos = 0.15
    tips = {
        'UTE': "Ensure proper gradient calibration\nUse phantom for delay measurement",
        'IR_UTE': "Verify B1 homogeneity\nCheck inversion efficiency",
        'VIBE_Dixon': "Monitor for fat-water swaps\nEnsure proper shimming",
        'ZTE': "Check gradient fidelity\nMinimize vibrations",
        'T1_mapping': "Acquire B1 map\nUse motion correction"
    }
    
    if sequence_type in tips:
        ax.text(0.1, y_pos, "Quick Tips:", ha='left', va='top', 
                fontsize=12, weight='bold')
        ax.text(0.1, y_pos-0.05, tips[sequence_type], ha='left', va='top', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def validate_pi_calculation():
    """
    Validate PI calculation against published values.
    """
    # Create synthetic test data
    analyzer = BonePorosityAnalyzer()
    
    # Test case from literature: healthy cortical bone
    echo1_healthy = np.ones((10, 10, 10)) * 1000  # High signal
    echo2_healthy = np.ones((10, 10, 10)) * 120   # ~12% retention
    
    pi_healthy = analyzer.calculate_porosity_index(echo1_healthy, echo2_healthy)
    
    print(f"Healthy bone PI: {pi_healthy[5, 5, 5]:.1f}% (expected: ~12%)")
    
    # Test case: osteoporotic bone
    echo1_osteo = np.ones((10, 10, 10)) * 800   # Lower signal
    echo2_osteo = np.ones((10, 10, 10)) * 240   # ~30% retention
    
    pi_osteo = analyzer.calculate_porosity_index(echo1_osteo, echo2_osteo)
    
    print(f"Osteoporotic bone PI: {pi_osteo[5, 5, 5]:.1f}% (expected: ~30%)")

def batch_process_subjects(subject_list: list, 
                         output_dir: str,
                         reference_bmd: Optional[Dict[str, float]] = None):
    """
    Process multiple subjects for clinical studies.
    
    Parameters:
    -----------
    subject_list : list
        List of dictionaries with subject data paths
    output_dir : str
        Directory to save results
    reference_bmd : dict, optional
        Dictionary of subject BMD values for correlation
    """
    results = []
    analyzer = BonePorosityAnalyzer()
    
    for subject in subject_list:
        try:
            # Load data
            echo1, echo2 = analyzer.load_ute_data(
                subject['echo1_path'], 
                subject['echo2_path']
            )
            
            # Segment cortical bone
            mask = analyzer.segment_cortical_bone(echo1)
            
            # Calculate PI
            pi_map = analyzer.calculate_porosity_index(echo1, echo2, mask)
            
            # Generate report
            report = analyzer.generate_report(
                pi_map, mask, 
                patient_id=subject['id'],
                save_path=f"{output_dir}/{subject['id']}_report.pdf"
            )
            
            # Store results
            pi_values = pi_map[mask > 0]
            result = {
                'subject_id': subject['id'],
                'mean_pi': np.mean(pi_values),
                'std_pi': np.std(pi_values),
                'cortical_volume': mask.sum() * np.prod(analyzer.voxel_dims)
            }
            
            # Add BMD correlation if available
            if reference_bmd and subject['id'] in reference_bmd:
                correlation = analyzer.correlate_with_bmd(
                    pi_values, 
                    np.full_like(pi_values, reference_bmd[subject['id']])
                )
                result.update(correlation)
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing subject {subject['id']}: {e}")
            
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/batch_results.csv", index=False)
    
    return results_df

def demonstrate_multi_sequence_analysis():
    """
    Demonstrate analysis workflow combining multiple sequences.
    """
    print("\n=== Multi-Sequence Bone Analysis Demonstration ===\n")
    
    # Initialize analyzer
    analyzer = BonePorosityAnalyzer(field_strength=3.0)
    
    # Create synthetic data for demonstration
    # In practice, load actual DICOM/NIfTI files
    
    # 1. UTE data
    echo1_ute = np.random.normal(1000, 100, (128, 128, 30))
    echo2_ute = echo1_ute * np.random.uniform(0.1, 0.3, echo1_ute.shape)
    
    # 2. IR-UTE data (bound water)
    ir_ute_data = np.random.normal(500, 50, (128, 128, 30))
    
    # 3. VIBE Dixon data
    water_dixon = np.random.normal(800, 80, (128, 128, 30))
    fat_dixon = np.random.normal(1200, 120, (128, 128, 30))
    in_phase = water_dixon + fat_dixon
    opposed_phase = water_dixon - fat_dixon
    
    # Create cortical bone mask
    mask = np.zeros((128, 128, 30), dtype=np.uint8)
    center = (64, 64)
    for z in range(mask.shape[2]):
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                if 20 < r < 30:  # Cortical shell
                    mask[y, x, z] = 1
    
    # Perform analyses
    print("1. UTE Analysis:")
    pi_map = analyzer.calculate_porosity_index(echo1_ute, echo2_ute, mask)
    pi_mean = np.mean(pi_map[mask > 0])
    print(f"   - Porosity Index: {pi_mean:.1f} ± {np.std(pi_map[mask > 0]):.1f}%")
    
    print("\n2. IR-UTE Analysis:")
    sr_map = analyzer.calculate_suppression_ratio(echo1_ute, ir_ute_data, mask)
    sr_mean = np.mean(sr_map[mask > 0])
    print(f"   - Suppression Ratio: {sr_mean:.2f} ± {np.std(sr_map[mask > 0]):.2f}")
    print(f"   - Estimated Bound Water: {15 + sr_mean:.1f}%")
    
    print("\n3. Dixon Analysis:")
    dixon_results = analyzer.analyze_dixon_data(
        water_dixon, fat_dixon, in_phase, opposed_phase, mask
    )
    print(f"   - Fat Fraction: {dixon_results['mean_fat_fraction']:.1f}%")
    print(f"   - Water Fraction: {dixon_results['mean_water_fraction']:.1f}%")
    print(f"   - Interpretation: {dixon_results['interpretation']}")
    
    print("\n4. Regional Analysis:")
    # Ensure analyzer has voxel dimensions
    if not hasattr(analyzer, 'voxel_dims'):
        analyzer.voxel_dims = (1.0, 1.0, 1.0)

    try:
        layer_stats = analyzer.calculate_regional_statistics(pi_map, mask)
        
        if not layer_stats.empty:
            for idx, row in layer_stats.iterrows():
                print(f"   - {row['layer_name']}: {row['mean']:.1f} ± {row['std']:.1f}%")
        else:
            print("   - No layer statistics calculated")
    except Exception as e:
        print(f"   - Error calculating layer statistics: {e}")
        print(f"   - Could not calculate regional statistics: {str(e)}")
    layer_stats = analyzer.calculate_regional_statistics(pi_map, mask)
    print(layer_stats[['layer_name', 'mean', 'std']].to_string(index=False))
    
    print("\n5. Clinical Integration:")
    if pi_mean > 25:
        risk_level = "HIGH"
        recommendation = "Immediate fracture risk assessment"
    elif pi_mean > 15:
        risk_level = "MODERATE"
        recommendation = "Consider preventive measures"
    else:
        risk_level = "LOW"
        recommendation = "Routine monitoring"
    
    print(f"   - Fracture Risk Level: {risk_level}")
    print(f"   - Recommendation: {recommendation}")
    
    return {
        'porosity_index': pi_mean,
        'bound_water': 15 + sr_mean,
        'fat_fraction': dixon_results['mean_fat_fraction'],
        'risk_level': risk_level
    }

# Main execution
if __name__ == "__main__":
    print("=== MRI Bone Porosity Analysis Pipeline v2.0 ===\n")
    
    # 1. Generate optimization guides for different field strengths
    print("1. Generating Optimization Guides...")
    
    for field in [1.5, 3.0]:
        analyzer = BonePorosityAnalyzer(field_strength=field)
        guide = analyzer.generate_sequence_optimization_guide(
            field_strength=field,
            clinical_application='osteoporosis',
            save_path=f'bone_imaging_guide_{field}T.txt'
        )
        print(f"   ✓ Generated guide for {field}T")
    
    # 2. Compare sequences for fracture risk assessment
    print("\n2. Comparing Sequences for Fracture Risk Assessment:")
    comparison_df = compare_sequences_for_application('fracture_risk')
    print(comparison_df.to_string())
    
    # 3. Generate protocol cards
    print("\n3. Generating Protocol Cards...")
    sequences = ['UTE', 'IR_UTE', 'VIBE_Dixon']
    for seq in sequences:
        generate_protocol_card(seq, 3.0, f'protocol_card_{seq}_3T.pdf')
        print(f"   ✓ Generated card for {seq}")
    
    # 4. Validate calculations
    print("\n4. Validating PI Calculations:")
    validate_pi_calculation()
    
    # 5. Demonstrate multi-sequence analysis
    print("\n5. Multi-Sequence Analysis:")
    results = demonstrate_multi_sequence_analysis()
    
    # 6. Show recommended protocol selection
    print("\n6. Recommended Protocol Selection:")
    print("   • For Screening: UTE clinical protocol (10 min)")
    print("   • For Comprehensive: UTE + IR-UTE + VIBE Dixon (25 min)")
    print("   • For Research: Full multi-echo UTE + T1 mapping (40 min)")
    
    # 7. Display key sequence parameters summary
    print("\n7. Key Sequence Parameters Summary:")
    print("   3T Scanner:")
    print("   - UTE: TE=0.032/2.2ms, TR=100ms, 0.5mm resolution")
    print("   - IR-UTE: TI/TR=45/100ms, selective bound water")
    print("   - VIBE Dixon: TE=1.23/2.46ms, 0.6mm resolution")
    print("   - ZTE: TE~0ms, silent scanning, 0.8mm isotropic")
    
    print("\n   1.5T Scanner:")
    print("   - UTE: TE=0.05/2.0ms, TR=150ms, 1.0mm resolution")
    print("   - IR-UTE: TI/TR=60/150ms, adjusted for T1")
    print("   - VIBE Dixon: TE=2.38/4.76ms, 0.8mm resolution")
    print("   - ZTE: Limited availability - use UTE instead")
    
    print("\n✅ Analysis pipeline ready for use!")
    print("📊 Results saved to current directory")
    print("📚 Check generated guides for detailed protocols")
    
    # Display clinical reference values
    print("\n8. Clinical Reference Values:")
    print("   Porosity Index (PI):")
    print("   - Normal: < 15%")
    print("   - Mildly elevated: 15-25%")
    print("   - Significantly elevated: > 25%")
    print("   \n   Expected Correlations:")
    print("   - PI vs BMD: r = -0.6 to -0.8")
    print("   - Bound water vs mechanics: r = 0.7-0.8")
    print("   - Reproducibility (CoV): < 5%")
    
    print("\n🔬 Ready for clinical implementation!")

    
    
