"""
Core Reporting Classes
======================

Defines the object hierarchy for the reporting system.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Dict
from datetime import datetime
import uuid
import base64
import io
import json

import pandas as pd
import numpy as np

from .engine import render_template
from coco_pipe.viz.plotly_utils import plot_embedding_interactive, plot_loss_history_interactive, plot_scree_interactive

class Element(ABC):
    """Abstract base class for all report elements."""
    
    @abstractmethod
    def render(self) -> str:
        """Render the element to HTML."""
        pass

class HtmlElement(Element):
    """
    Wrapper for raw HTML content.
    
    Parameters
    ----------
    html : str
        The raw HTML string to include.
    """
    def __init__(self, html: str):
        self.html = html
        
    def render(self) -> str:
        return self.html

class ImageElement(Element):
    """
    Embeds an image or matplotlib figure as Base64.
    
    Parameters
    ----------
    src : str, bytes, Path, or matplotlib.figure.Figure
        The image source.
    caption : str, optional
        Caption text for the figure.
    width : str, optional
        CSS width (e.g., '100%', '600px'). Default '100%'.
    """
    def __init__(self, src: Any, caption: Optional[str] = None, width: str = "100%"):
        self.src = src
        self.caption = caption
        self.width = width
        
    def _encode_image(self) -> str:
        """Convert input to base64 string."""
        # Check for Matplotlib Figure
        if hasattr(self.src, 'savefig'):
            buf = io.BytesIO()
            self.src.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            buf.seek(0)
            data = buf.read()
            return base64.b64encode(data).decode('utf-8')
        
        # Check for bytes
        if isinstance(self.src, bytes):
            return base64.b64encode(self.src).decode('utf-8')
            
        # Check for path (str or Path)
        if isinstance(self.src, (str, Path)):
            import pathlib
            p = pathlib.Path(self.src)
            if p.exists():
                return base64.b64encode(p.read_bytes()).decode('utf-8')
        
        raise ValueError(f"Unsupported image source type: {type(self.src)}")

    def render(self) -> str:
        b64_str = self._encode_image()
        html = f'''
        <figure class="my-6">
            <img src="data:image/png;base64,{b64_str}" style="width: {self.width};" class="rounded shadow-sm mx-auto border border-gray-100">
            {f'<figcaption class="text-center text-sm text-gray-500 mt-2">{self.caption}</figcaption>' if self.caption else ''}
        </figure>
        '''
        return html

class PlotlyElement(Element):
    """
    Embeds a Plotly figure using lazy loading.
    
    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to render.
    height : str, optional
        Height of the plot plot container. Default "500px".
    """
    def __init__(self, figure: Any, height: str = "500px"):
        self.figure = figure
        self.height = height
        
    def render(self) -> str:
        fig_dict = self.figure.to_dict()
        json_str = json.dumps(fig_dict)
        
        # Safe escape for HTML attribute
        safe_json = json_str.replace('"', '&quot;')
        
        html = f'''
        <div class="my-6">
            <div class="lazy-plot w-full rounded shadow-sm border border-gray-100 bg-gray-50 flex items-center justify-center text-gray-400 animate-pulse" 
                 style="height: {self.height};"
                 data-figure="{safe_json}">
                 <span class="sr-only">Loading Plot...</span>
            </div>
        </div>
        '''
        return html

class TableElement(Element):
    """
    Renders a Pandas DataFrame or Dict as a styled HTML table.
    
    Parameters
    ----------
    data : DataFrame, Dict, or List[Dict]
        Data to display.
    title : str, optional
        Title describing the table.
    """
    def __init__(self, data: Any, title: Optional[str] = None):
        self.data = data
        self.title = title
        
    def render(self) -> str:
        # Convert to DataFrame
        if isinstance(self.data, pd.DataFrame):
            df = self.data
        else:
            df = pd.DataFrame(self.data)
            
        # Basic Tailwind Styling
        html = f'<div class="overflow-x-auto my-4">'
        if self.title:
            html += f'<h4 class="text-sm font-semibold text-gray-700 mb-2 uppercase tracking-wide">{self.title}</h4>'
            
        # Render Table
        # We manually render to control classes better than to_html
        html += '<table class="min-w-full divide-y divide-gray-200 border text-sm">'
        
        # Header
        html += '<thead class="bg-gray-50"><tr>'
        for col in df.columns:
            html += f'<th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{col}</th>'
        html += '</tr></thead>'
        
        # Body
        html += '<tbody class="bg-white divide-y divide-gray-200">'
        for _, row in df.iterrows():
            html += '<tr>'
            for val in row:
                html += f'<td class="px-4 py-3 whitespace-nowrap text-gray-700">{val}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'
        
        return html

class ContainerElement(Element):
    """
    Base class for elements that contain other elements.
    """
    def __init__(self):
        self.children: List[Element] = []
        
    def add_element(self, element: Union[Element, str]):
        """
        Add a child element.
        
        Parameters
        ----------
        element : Element or str
            The element to add. specific strings are converted to HtmlElement.
        """
        if isinstance(element, str):
            element = HtmlElement(element)
        self.children.append(element)
        return self # Fluent interface
        
    def render_children(self) -> str:
        """Render all child elements concatenated."""
        return "\n".join([c.render() for c in self.children])
        
    def render(self) -> str:
        return self.render_children()

class Section(ContainerElement):
    """
    A logical section of the report.
    
    Parameters
    ----------
    title : str
        The section title.
    icon : str, optional
        SVG icon or emoji to display next to the title.
    """
    def __init__(self, title: str, icon: Optional[str] = None):
        super().__init__()
        self.title = title
        self.icon = icon
        
    def render(self) -> str:
        content = self.render_children()
        return render_template("section.html", title=self.title, icon=self.icon, content=content)

class Report(ContainerElement):
    """
    The main report container.
    
    Parameters
    ----------
    title : str
        The report title.

    Examples
    --------
    >>> report = Report("My Analysis")
    >>> report.add_markdown("## Introduction\\nThis is a report.")
    >>> report.save("report.html")
    """
    def __init__(self, title: str = "CoCo Analysis Report"):
        super().__init__()
        self.title = title
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
    def add_markdown(self, text: str) -> 'Report':
        """
        Add a markdown block to the report.
        
        Note: Requires 'markdown' package. If not present, falls back to raw text in <pre>.
        """
        try:
            import markdown
            html = markdown.markdown(text, extensions=['extra'])
            # Wrap in prose class for consistent styling
            wrapper = f'<div class="prose prose-sm max-w-none text-gray-700">{html}</div>'
            self.add_element(HtmlElement(wrapper))
        except ImportError:
            # Fallback
            safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
            html = f'<div class="whitespace-pre-wrap font-mono text-sm bg-gray-50 p-4 rounded">{safe_text}</div>'
            self.add_element(HtmlElement(html))
        return self

    def add_section(self, section: Section) -> 'Report':
        """Syntactic sugar for adding a Section."""
        return self.add_element(section)

    def add_figure(self, fig: Any, caption: Optional[str] = None) -> 'Report':
        """
        Add a figure (Matplotlib) or Image.
        """
        self.add_element(ImageElement(fig, caption=caption))
        return self

    def add_container(self, container: Any, name: str = "Data Overview") -> 'Report':
        """
        Add a summary section for a DataContainer.
        
        Parameters
        ----------
        container : DataContainer
            The data container to summarize.
        name : str
            Title for the section.
        """
        # Create Section
        sec = Section(title=name, icon="💾")
        
        # 1. Metadata Table
        # Extract general info
        meta_info = {
            "Dimension": list(container.dims),
            "Size": list(container.shape),
        }
        
        # Add shape info to meta_info for table
        # We'll pivot this for cleaner display: Dim Name -> Size
        dims_data = [{"Dimension": d, "Size": s} for d, s in zip(container.dims, container.shape)]
        
        sec.add_element(TableElement(dims_data, title="Dimensions"))
        
        # Coordinates Info
        if container.coords:
            coords_data = [{"Name": k, "Type": str(np.array(v).dtype), "Count": len(v)} for k, v in container.coords.items()]
            sec.add_element(TableElement(coords_data, title="Coordinates"))
            
        # 2. Simple Distribution Plot (if applicable)
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 3))
            
            # Check for 'y' (Class Distribution)
            if container.y is not None:
                # Plot Class counts
                y_series = pd.Series(container.y)
                y_series.value_counts().plot(kind='bar', ax=ax, color='skyblue')
                ax.set_title("Class Distribution")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                caption = "Target label distribution."
                
            else:
                # Plot simple data histogram (sampled if large)
                data_flat = container.X.flatten()
                if len(data_flat) > 5000:
                    data_flat = np.random.choice(data_flat, 5000, replace=False)
                
                ax.hist(data_flat, bins=30, color='gray', alpha=0.7)
                ax.set_title("Data Value Distribution (Sampled)")
                caption = "Histogram of data values."
                
            plt.tight_layout()
            sec.add_element(ImageElement(fig, caption=caption, width="80%"))
            plt.close(fig)
            
        except Exception as e:
            sec.add_element(HtmlElement(f"<div class='text-red-500 text-xs'>Could not generate plot: {e}</div>"))

        self.add_section(sec)
        return self

    def add_reduction(self, reducer: Any, name: str = None) -> 'Report':
        """
        Add a dimensionality reduction result to the report.
        
        Parameters
        ----------
        reducer : DimReduction or similar
            Fitted dimensionality reduction object. Must have 'embedding_' attribute.
        name : str, optional
            Name of the reduction (e.g., "UMAP"). If None, inferred from class name.
        """
        if name is None:
            name = type(reducer).__name__
            
        sec = Section(title=name, icon="📉")
        
        # 1. Main Interactive Embedding Plot
        if hasattr(reducer, 'embedding_'):
            emb = reducer.embedding_
            dims = emb.shape[1]
            if dims > 3: dimensions = 3
            else: dimensions = dims
            
            fig = plot_embedding_interactive(
                embedding=emb, 
                title=f"{name} Embedding",
                dimensions=dimensions
            )
            sec.add_element(PlotlyElement(fig))
            
        # 2. Metrics Table
        # (Future Implementation: Compute or display cached metrics if available)
        # currently skipped to avoid re-computation overhead without original data.
        
        # 3. Diagnostics
        # Loss Curve
        if hasattr(reducer, 'loss_history_') and reducer.loss_history_ is not None:
             fig_loss = plot_loss_history_interactive(reducer.loss_history_)
             sec.add_element(PlotlyElement(fig_loss, height="350px"))
             
        # Scree Plot (PCA)
        if hasattr(reducer, 'explained_variance_ratio_'):
            fig_scree = plot_scree_interactive(reducer.explained_variance_ratio_)
            sec.add_element(PlotlyElement(fig_scree, height="350px"))

        self.add_section(sec)
        return self

    def render(self) -> str:
        """
        Render the full HTML report.
        """
        # Get content from children (Sections)
        content_html = super().render()
        
        # Wrap in base template
        return render_template(
            "base.html", 
            title=self.title, 
            content=content_html,
            timestamp=self.timestamp
        )

    def save(self, filename: str) -> None:
        """
        Render and save the report to a file.
        
        Parameters
        ----------
        filename : str
            Path to save the HTML file.
        """
        full_html = self.render()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_html)
