class CanonSequenceWidget(QWidget):
    """Widget for displaying a single sequence in Canon style"""
    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)
        
        # Set background
        self.setStyleSheet("""
            CanonSequenceWidget {
                background-color: #1a2332;
                border: 1px solid #2c4158;
                border-radius: 5px;
            }
        """)
        
        # Sequence header
        self.create_header()
        
        # Parameter sections
        self.create_parameter_sections()
        
    def create_header(self):
        """Create sequence header with name and type"""
        header_widget = QWidget()
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(5, 5, 5, 5)
        header_layout.setSpacing(3)
        header_widget.setLayout(header_layout)
        
        # Top row with main info
        top_row = QHBoxLayout()
        
        # Sequence number badge
        number_label = QLabel(str(self.sequence.number))
        number_label.setStyleSheet("""
            background-color: #4a90e2;
            color: #ffffff;
            font-size: 12px;
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 3px;
            max-width: 30px;
        """)
        number_label.setAlignment(Qt.AlignCenter)
        top_row.addWidget(number_label)
        
        # Sequence name and type
        name_layout = QVBoxLayout()
        name_layout.setSpacing(2)
        
        # Main name
        seq_name_label = QLabel(self.sequence.name)
        seq_name_label.setStyleSheet("""
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            padding-left: 5px;
        """)
        name_layout.addWidget(seq_name_label)
        
        # Sequence type if different from name
        if self.sequence.sequence_type != "Unknown" and self.sequence.sequence_type not in self.sequence.name:
            type_label = QLabel(f"Type: {self.sequence.sequence_type}")
            type_label.setStyleSheet("""
                color: #a0a0a0;
                font-size: 11px;
                padding-left: 5px;
            """)
            name_layout.addWidget(type_label)
            
        top_row.addLayout(name_layout)
        top_row.addStretch()
        
        # Right side controls
        # Enable/disable checkbox
        self.enable_check = QCheckBox("Enabled")
        self.enable_check.setChecked(self.sequence.enabled)
        self.enable_check.setStyleSheet("""
            QCheckBox {
                color: #e0e0e0;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        top_row.addWidget(self.enable_check)
        
        # Time display
        if 'TACQT' in self.sequence.parameters:
            time_label = QLabel(f"Time: {self.sequence.parameters['TACQT'].value}")
            time_label.setStyleSheet("""
                color: #4a90e2;
                font-size: 11px;
                font-weight: bold;
                margin-left: 15px;
            """)
            top_row.addWidget(time_label)
            
        # Key parameters summary (TR/TE)
        summary_params = []
        if 'TR' in self.sequence.parameters:
            summary_params.append(f"TR: {self.sequence.parameters['TR'].value}ms")
        if 'TE' in self.sequence.parameters:
            summary_params.append(f"TE: {self.sequence.parameters['TE'].value}ms")
            
        if summary_params:
            summary_label = QLabel(" | ".join(summary_params))
            summary_label.setStyleSheet("""
                color: #e0e0e0;
                font-size: 10px;
                margin-left: 15px;
            """)
            top_row.addWidget(summary_label)
        
        header_layout.addLayout(top_row)
        
        # Header background
        header_widget.setStyleSheet("""
            background-color: #2c4158;
            border-radius: 3px;
        """)
        
        self.layout().addWidget(header_widget)
        
    def create_parameter_sections(self):
        """Create parameter sections with compact grid layout"""
        # Group parameters by category
        categories = {}
        for param_key, param in self.sequence.parameters.items():
            category = param.category
            if category not in categories:
                categories[category] = []
            categories[category].append((param_key, param))
        
        # Create sections for each category
        category_order = ['Basic', 'Geometry', 'Acquisition', 'Advanced', 'Info']
        
        # Add any categories not in the standard order
        for cat in sorted(categories.keys()):
            if cat not in category_order:
                category_order.append(cat)
        
        for category in category_order:
            if category in categories:
                self.create_category_section(category, categories[category])
                
    def create_category_section(self, category_name, parameters):
        """Create a section for a parameter category"""
        # Don't create empty sections
        if not parameters:
            return
            
        # Category frame
        category_frame = QFrame()
        category_frame.setStyleSheet("""
            QFrame {
                background-color: #1a2332;
                border: 1px solid #2c4158;
                border-radius: 3px;
                padding: 5px;
                margin: 2px;
            }
        """)
        
        frame_layout = QVBoxLayout()
        frame_layout.setContentsMargins(5, 5, 5, 5)
        frame_layout.setSpacing(3)
        category_frame.setLayout(frame_layout)
        
        # Category label (except for Basic)
        if category_name != "Basic":
            cat_label = QLabel(category_name)
            cat_label.setStyleSheet("""
                color: #a0a0a0;
                font-size: 10px;
                font-weight: bold;
                margin-bottom: 3px;
            """)
            frame_layout.addWidget(cat_label)
        
        # Parameter grid
        param_grid = QGridLayout()
        param_grid.setSpacing(8)
        param_grid.setContentsMargins(0, 0, 0, 0)
        
        # Add parameters in grid (3-4 per row for compact display)
        row = 0
        col = 0
        max_cols = 4 if category_name in ['Basic', 'Geometry'] else 3
        
        for param_key, param in parameters:
            param_widget = CanonCompactParameter(
                param.display_name,
                param.value,
                param.unit
            )
            param_grid.addWidget(param_widget, row, col)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
                
        frame_layout.addLayout(param_grid)
        self.layout().addWidget(category_frame)
