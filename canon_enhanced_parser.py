"""
Enhanced Canon XML Parser with improved sequence detection and naming
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import os
import re

class EnhancedCanonXMLParser:
    """Enhanced parser that better detects all sequences in Canon PAS XML files"""
    
    def __init__(self):
        self.param_definitions = PARAMETER_DEFINITIONS
        self.debug_mode = True  # Enable to see what sequences are found
        
    def parse_file(self, file_path: str) -> CanonProtocol:
        """Parse a Canon PAS XML file with better sequence detection"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract protocol name and metadata
            protocol_name = self._extract_protocol_name(root, file_path)
            metadata = self._parse_metadata(root)
            
            # Parse all sequences with multiple detection methods
            sequences = self._find_all_sequences(root)
            
            if self.debug_mode:
                print(f"Found {len(sequences)} sequences in {os.path.basename(file_path)}")
                for seq in sequences:
                    print(f"  - {seq.get_display_name()}")
            
            return CanonProtocol(
                name=protocol_name,
                filename=file_path,
                sequences=sequences,
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"Error parsing Canon XML file: {str(e)}")
    
    def _find_all_sequences(self, root: ET.Element) -> List[MRISequence]:
        """Find all sequences using multiple detection strategies"""
        sequences = []
        sequence_number = 1
        
        # Strategy 1: Look for explicit sequence containers
        seq_containers = [
            './/Sequence', './/SEQUENCE', './/sequence',
            './/Series', './/SERIES', './/series',
            './/Scan', './/SCAN', './/scan',
            './/Study', './/STUDY', './/study'
        ]
        
        for container_path in seq_containers:
            seq_elements = root.findall(container_path)
            if seq_elements:
                for seq_elem in seq_elements:
                    seq = self._parse_sequence_element(seq_elem, sequence_number)
                    if seq and self._has_valid_parameters(seq):
                        sequences.append(seq)
                        sequence_number += 1
                        
        # Strategy 2: Look for sequence groups by common patterns
        if not sequences:
            sequences = self._find_sequences_by_pattern(root, sequence_number)
            
        # Strategy 3: If still no sequences, check if entire document is one sequence
        if not sequences:
            single_seq = self._parse_single_sequence(root)
            if self._has_valid_parameters(single_seq):
                sequences = [single_seq]
                
        return sequences
    
    def _find_sequences_by_pattern(self, root: ET.Element, start_number: int) -> List[MRISequence]:
        """Find sequences by looking for repeating parameter patterns"""
        sequences = []
        sequence_number = start_number
        
        # Look for elements that might indicate sequence boundaries
        # Canon often uses numbered or indexed elements
        patterns = [
            (r'SEQ(\d+)', 'SEQ'),
            (r'SERIES(\d+)', 'SERIES'),
            (r'SCAN(\d+)', 'SCAN'),
            (r'.*_(\d+)$', 'indexed')
        ]
        
        # Collect all elements that might be sequence indicators
        all_elements = root.findall('.//*')
        sequence_groups = {}
        
        for elem in all_elements:
            if elem.tag:
                for pattern, prefix in patterns:
                    match = re.match(pattern, elem.tag)
                    if match:
                        seq_index = match.group(1)
                        key = f"{prefix}_{seq_index}"
                        if key not in sequence_groups:
                            sequence_groups[key] = []
                        sequence_groups[key].append(elem)
                        
        # Parse each group as a potential sequence
        for group_key, elements in sequence_groups.items():
            # Create a virtual container for these elements
            virtual_container = ET.Element("VirtualSequence")
            for elem in elements:
                virtual_container.append(elem)
                
            seq = self._parse_sequence_element(virtual_container, sequence_number)
            if seq and self._has_valid_parameters(seq):
                sequences.append(seq)
                sequence_number += 1
                
        return sequences
    
    def _parse_sequence_element(self, seq_elem: ET.Element, number: int) -> Optional[MRISequence]:
        """Parse a single sequence element with better name extraction"""
        # Try multiple ways to get sequence identification
        seq_name = self._extract_sequence_name(seq_elem)
        seq_type = self._extract_sequence_type(seq_elem)
        seq_comment = self._extract_sequence_comment(seq_elem)
        
        # Build a meaningful display name
        if seq_comment and seq_comment != seq_name:
            display_name = seq_comment
        elif seq_name and seq_name != "Unknown":
            display_name = seq_name
        else:
            display_name = f"Sequence {number}"
            
        # Parse parameters for this sequence
        parameters = self._parse_parameters(seq_elem)
        
        # Check if sequence is enabled
        enabled = self._extract_text(seq_elem, './/ENABLED', 'TRUE') == 'TRUE'
        
        return MRISequence(
            number=number,
            name=display_name,
            sequence_type=seq_type,
            parameters=parameters,
            enabled=enabled
        )
    
    def _extract_sequence_name(self, elem: ET.Element) -> str:
        """Extract sequence name from various possible fields"""
        name_fields = [
            './/SEQNAME', './/SequenceName', './/Name', './/NAME',
            './/SeriesDescription', './/SERIESDESCRIPTION',
            './/ProtocolName', './/PROTOCOLNAME',
            './/Description', './/DESCRIPTION',
            './/Label', './/LABEL'
        ]
        
        for field in name_fields:
            name = self._extract_text(elem, field)
            if name and name != "Unknown":
                return name
                
        # Try to get from attributes
        if 'name' in elem.attrib:
            return elem.attrib['name']
        if 'Name' in elem.attrib:
            return elem.attrib['Name']
            
        return "Unknown"
    
    def _extract_sequence_type(self, elem: ET.Element) -> str:
        """Extract sequence type/ID from various possible fields"""
        type_fields = [
            './/SEQID', './/SequenceID', './/ID',
            './/SequenceType', './/SEQTYPE', './/Type', './/TYPE',
            './/PulseSequenceName', './/PULSESEQUENCENAME'
        ]
        
        for field in type_fields:
            seq_type = self._extract_text(elem, field)
            if seq_type:
                return seq_type
                
        return "Unknown"
    
    def _extract_sequence_comment(self, elem: ET.Element) -> str:
        """Extract sequence comment or additional description"""
        comment_fields = [
            './/Comment', './/COMMENT', './/Comments', './/COMMENTS',
            './/UserComment', './/USERCOMMENT',
            './/Note', './/NOTE', './/Notes', './/NOTES',
            './/Annotation', './/ANNOTATION'
        ]
        
        for field in comment_fields:
            comment = self._extract_text(elem, field)
            if comment:
                return comment
                
        return ""
    
    def _extract_protocol_name(self, root: ET.Element, file_path: str) -> str:
        """Extract protocol name from XML or filename"""
        # Try to get from XML
        protocol_name = self._extract_text(root, './/ProtocolName')
        if not protocol_name:
            protocol_name = self._extract_text(root, './/PROTOCOLNAME')
        if not protocol_name:
            protocol_name = self._extract_text(root, './/Name')
        if not protocol_name:
            protocol_name = self._extract_text(root, './/NAME')
            
        # Fall back to filename
        if not protocol_name:
            protocol_name = os.path.basename(file_path).replace('.xml', '')
            
        return protocol_name
    
    def _has_valid_parameters(self, sequence: MRISequence) -> bool:
        """Check if sequence has valid imaging parameters"""
        # A valid sequence should have at least some basic parameters
        required_params = ['TR', 'TE', 'FOV', 'MTX']
        found_params = sum(1 for param in required_params if param in sequence.parameters)
        return found_params >= 2  # At least 2 of the basic parameters
    
    def _parse_parameters(self, element: ET.Element) -> Dict[str, SequenceParameter]:
        """Parse all parameters from an element"""
        parameters = {}
        
        # First, try to get parameters by known names
        for param_key, param_def in self.param_definitions.items():
            # Try different XML paths and variations
            value = None
            paths_to_try = [
                f'.//{param_key}',
                f'./{param_key}',
                f'.//{param_key.lower()}',
                f'.//{param_key.upper()}',
                f'.//Param[@name="{param_key}"]',
                f'.//Parameter[@name="{param_key}"]',
                f'.//PARAM[@name="{param_key}"]'
            ]
            
            for path in paths_to_try:
                if path.contains('@'):
                    # XPath with attribute
                    elem = element.find(path)
                    if elem is not None:
                        value = elem.get('value', elem.text)
                else:
                    value = self._extract_text(element, path)
                    
                if value is not None:
                    break
            
            if value is not None:
                # Create parameter
                param = SequenceParameter(
                    name=param_def.get('name', param_key),
                    value=self._convert_value(value, param_def.get('type', 'str')),
                    unit=param_def.get('unit', ''),
                    category=param_def.get('category', 'Basic'),
                    display_name=param_def.get('display', param_key)
                )
                
                parameters[param_key] = param
        
        # Also look for any unknown parameters
        self._find_additional_parameters(element, parameters)
        
        # Handle special formatting
        self._format_special_parameters(parameters)
        
        return parameters
    
    def _find_additional_parameters(self, element: ET.Element, parameters: Dict[str, SequenceParameter]):
        """Find parameters that aren't in our predefined list"""
        # Look for common parameter container patterns
        param_containers = element.findall('.//Parameter') + element.findall('.//Param') + element.findall('.//PARAM')
        
        for param_elem in param_containers:
            name = param_elem.get('name') or param_elem.get('Name') or param_elem.get('NAME')
            value = param_elem.get('value') or param_elem.get('Value') or param_elem.text
            
            if name and value and name not in parameters:
                # Add as unknown parameter
                param = SequenceParameter(
                    name=name,
                    value=value,
                    unit='',
                    category='Advanced',
                    display_name=name
                )
                parameters[name] = param
    
    def _format_special_parameters(self, parameters: Dict[str, SequenceParameter]):
        """Format special parameters like FOV and Matrix"""
        if 'FOV' in parameters:
            fov_value = str(parameters['FOV'].value)
            if 'x' not in fov_value and len(fov_value) >= 4 and fov_value.isdigit():
                # Convert "2424" to "24x24"
                half = len(fov_value) // 2
                parameters['FOV'].value = f"{fov_value[:half]}x{fov_value[half:]}"
        
        if 'MTX' in parameters:
            mtx_value = str(parameters['MTX'].value)
            if 'x' not in mtx_value and len(mtx_value) >= 6 and mtx_value.isdigit():
                # Convert "256256" to "256x256"
                half = len(mtx_value) // 2
                parameters['MTX'].value = f"{mtx_value[:half]}x{mtx_value[half:]}"
    
    def _parse_single_sequence(self, root: ET.Element) -> MRISequence:
        """Parse the entire document as a single sequence (fallback)"""
        seq_name = self._extract_sequence_name(root)
        seq_type = self._extract_sequence_type(root)
        
        if seq_name == "Unknown":
            seq_name = "Main Sequence"
        
        parameters = self._parse_parameters(root)
        
        return MRISequence(
            number=1,
            name=seq_name,
            sequence_type=seq_type,
            parameters=parameters,
            enabled=True
        )
    
    def _parse_metadata(self, root: ET.Element) -> Dict[str, str]:
        """Parse metadata from XML"""
        metadata = {}
        
        metadata_paths = {
            'StudyDate': ['.//StudyDate', './/STUDYDATE'],
            'StudyTime': ['.//StudyTime', './/STUDYTIME'],
            'PatientName': ['.//PatientName', './/PATIENTNAME'],
            'PatientID': ['.//PatientID', './/PATIENTID'],
            'Modality': ['.//Modality', './/MODALITY'],
            'Manufacturer': ['.//Manufacturer', './/MANUFACTURER'],
            'InstitutionName': ['.//InstitutionName', './/INSTITUTIONNAME'],
            'StationName': ['.//StationName', './/STATIONNAME'],
            'ProtocolName': ['.//ProtocolName', './/PROTOCOLNAME'],
            'SeriesDescription': ['.//SeriesDescription', './/SERIESDESCRIPTION'],
        }
        
        for key, paths in metadata_paths.items():
            for path in paths:
                value = self._extract_text(root, path)
                if value:
                    metadata[key] = value
                    break
                    
        return metadata
    
    def _extract_text(self, element: ET.Element, xpath: str, default: Optional[str] = None) -> Optional[str]:
        """Extract text from XML element"""
        found_elem = element.find(xpath)
        if found_elem is not None:
            if found_elem.text:
                return found_elem.text.strip()
            # Also check for value attribute
            if 'value' in found_elem.attrib:
                return found_elem.attrib['value'].strip()
        return default
    
    def _convert_value(self, value: str, value_type: str) -> Any:
        """Convert string value to appropriate type"""
        try:
            if value_type == 'int':
                return int(float(value))  # Handle decimal strings
            elif value_type == 'float':
                return float(value)
            else:
                return value
        except:
            return value

# Update the MRISequence class to have a better display name method
@dataclass
class MRISequence:
    """Represents a single sequence within a protocol"""
    number: int
    name: str
    sequence_type: str
    parameters: Dict[str, SequenceParameter]
    enabled: bool = True
    
    def get_display_name(self) -> str:
        """Get formatted display name for the sequence"""
        # If we have a good name, use it with the type
        if self.name and self.name != "Unknown" and self.name != f"Sequence {self.number}":
            if self.sequence_type and self.sequence_type != "Unknown":
                return f"{self.name} [{self.sequence_type}]"
            return self.name
        # Otherwise fall back to type with number
        elif self.sequence_type and self.sequence_type != "Unknown":
            return f"{self.sequence_type} (Seq {self.number})"
        # Last resort
        return f"Sequence {self.number}"
    
    def get_short_name(self) -> str:
        """Get a shorter name for compact displays"""
        if self.name and self.name != "Unknown":
            # Truncate long names
            if len(self.name) > 30:
                return self.name[:27] + "..."
            return self.name
        return self.sequence_type
