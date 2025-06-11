import gradio as gr
from typing_extensions import Any, Dict, List
from gradio.components import Component

class MetadataForm:
    """Dynamic form component class for creating metadata entry fields based on database schema.
    This class provides a structured interface for collecting metadata information about media content
    including file paths, content type, tools used, tasks, timing information, and viewing perspective.
    The form is built using Gradio components and is designed for content management and annotation workflows.
    Attributes:
        components (Dict[str, Component]): Dictionary containing all form components keyed by field name
    Methods:
        _create_components(): Creates and configures all form components with appropriate labels and info
        get_component(name): Retrieves a specific form component by its field name
        get_all_components(): Returns the complete dictionary of form components
    Form Fields:
        - remote_path: Text input for file location/identifier
        - modality: Dropdown for content type (video, audio, text, document)
        - toollist: Text input for comma-separated list of tools/equipment
        - task: Text input for task or learning objective description
        - t_start: Number input for start timestamp in seconds
        - t_end: Number input for end timestamp in seconds
        - perspective: Text input for viewpoint or contextual perspective
    Example:
        ```python
        form = MetadataForm()
        modality_component = form.get_component('modality')
        all_components = form.get_all_components()
        ```
    """

    def __init__(self):
        self.components = {}
        self.rendered_components = {}
        self._populate_rendered_components()

    def _populate_rendered_components(self):
        """Populate rendered_components with default components on instantiation."""
        default_columns = ['remote_path', 'modality', 'toollist', 'task', 't_start', 't_end', 'perspective']
        self.create_dynamic_form(default_columns)
        self.rendered_components = self.components.copy()
    
    def create_dynamic_form(self, columns: List[str]) -> Dict[str, Component]:
        """Create form components based on database columns."""
        self.components = {}
        
        for col in columns:
            # Create appropriate input component based on column name/type
            if col in ['t_start', 't_end']:
                component = gr.Number(
                    label=col.replace('_', ' ').title(), 
                    value=0.0,
                    info=f"{col.replace('_', ' ')} in seconds"
                )
            elif col == 'modality':
                component = gr.Dropdown(
                    choices=["video", "audio", "text", "document"],
                    label=col.replace('_', ' ').title(),
                    info="Type of media content"
                )
            else:
                component = gr.Textbox(
                    label=col.replace('_', ' ').title(),
                    placeholder=f"Enter {col.replace('_', ' ')}",
                    info=f"Enter value for {col}"
                )
            
            self.components[col] = component
        
        return self.components
    
    def render_form(self, columns: List[str]):
        """Render all components for the given columns in current Gradio context."""
        self.create_dynamic_form(columns)
        self.rendered_components = {}
        
        for col, component in self.components.items():
            # Component is automatically rendered when created in Gradio context
            self.rendered_components[col] = component
        
        return self.rendered_components
    
    def get_component_values(self) -> List[Component]:
        """Get list of rendered components in order for event handlers."""
        return list(self.rendered_components.values())
    
    def get_component_keys(self) -> List[str]:
        """Get a list of rendered comoponents' names."""
        return list(self.rendered_components.keys())
    
    def get_component(self, name: str):
        """Get a specific component by name."""
        return self.rendered_components.get(name)
    
    def get_all_components(self):
        """Get all rendered components."""
        return self.rendered_components

