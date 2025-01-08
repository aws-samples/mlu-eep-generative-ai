import os
import json
import logging
import textwrap
from typing import List, Dict, Union, Tuple

import boto3
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from botocore.exceptions import ClientError
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from IPython.display import Markdown, display

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Allow PIL to load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def generate_text_mistral(model_id: str, body: str) -> Dict:
    """
    Generate text using a Mistral AI model through AWS Bedrock.
    
    Args:
        model_id: The identifier for the Mistral model
        body: JSON-formatted request body
        
    Returns:
        Dict containing the model's response
    """
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    return bedrock.invoke_model(body=body, modelId=model_id)

def invoke_mistral(
    prompt: str,
    model: str = 'mistral.mistral-7b-instruct-v0:2',
    temperature: float = 0.0,
    max_tokens: int = 1000,
    stop_sequences: List[str] = [],
    n: int = 1
) -> str:
    """
    Invoke Mistral model with specified parameters.
    
    Args:
        prompt: Input text for the model
        model: Model identifier
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
        stop_sequences: List of sequences where generation should stop
        n: Number of completions to generate
        
    Returns:
        Generated text from the model
    """
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 5
    })
    
    response = generate_text_mistral(model_id=model, body=body)
    response_body = json.loads(response.get('body').read())
    return response_body.get('outputs')[0]['text']

def generate_text_titan(model_id: str, body: str) -> Dict:
    """
    Generate text using Amazon Titan Text models.
    
    Args:
        model_id: The identifier for the Titan model
        body: JSON-formatted request body
        
    Returns:
        Dict containing the model's response
    """
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    
    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    
    response_body = json.loads(response.get("body").read())
    
    if response_body.get("error"):
        raise Exception(f"Text generation error: {response_body.get('error')}")
        
    return response_body

def invoke_titan(
    prompt: str,
    model: str = "amazon.titan-text-lite-v1",
    temperature: float = 0.0,
    max_tokens: int = 1000,
    stop_sequences: List[str] = [],
    n: int = 1
) -> str:
    """
    Invoke Titan model with specified parameters.
    
    Args:
        Similar to invoke_mistral
        
    Returns:
        Generated text from the model
    """
    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": max_tokens,
            "stopSequences": stop_sequences,
            "temperature": temperature,
            "topP": 0.9
        }
    })
    
    response_body = generate_text_titan(model, body)
    return response_body['results'][0]['outputText']

def run_inference(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 1000,
    stop_sequences: List[str] = [],
    n: int = 1
) -> List[str]:
    """
    Run inference with models hosted in Bedrock.
    
    Args:
        Similar to previous functions
        
    Returns:
        List of generated outputs
    """
    model_provider = model.split(".")[0]
    outputs = []
    
    for _ in range(n):
        if model_provider == "mistral":
            outputs.append(invoke_mistral(prompt, model, temperature, max_tokens, stop_sequences, n))
        else:
            outputs.append(invoke_titan(prompt, model, temperature, max_tokens, stop_sequences, n))
    
    return outputs

# Image Processing Functions

def process_images(folder_path: str, max_file_size: int = 5*1024*1024, max_resolution: Tuple[int, int] = (720, 720)) -> None:
    """
    Process images in a folder to ensure they meet size and resolution constraints.
    
    Args:
        folder_path: Path to folder containing images
        max_file_size: Maximum allowed file size in bytes
        max_resolution: Maximum allowed resolution as (width, height)
    """
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.png', 'jpeg')):
            continue
            
        image_path = os.path.join(folder_path, filename)
        
        with Image.open(image_path) as img:
            needs_resize = (
                os.path.getsize(image_path) > max_file_size or
                img.size > max_resolution
            )
            
            if needs_resize:
                img.thumbnail(max_resolution)
                img.save(image_path)
                logger.info(f"Resized {filename}")
            else:
                logger.info(f"{filename} is already compliant")

# Visualization Functions

def plot_images(dir_name: str) -> None:
    """
    Display multiple images in a grid layout.
    
    Args:
        dir_name: Directory containing the images
    """
    image_paths = [
        os.path.join(dir_name, f) 
        for f in os.listdir(dir_name) 
        if f.lower().endswith(('.jpg', '.png', 'jpeg'))
    ]
    
    num_images = len(image_paths)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 15))
    axes = axes.ravel()
    
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        ax = axes[i]
        ax.imshow(image)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
    
# Dimensionality Reduction and Visualization Functions

def reduce_dimensionality(array: np.ndarray, n_components: int = 2) -> PCA:
    """
    Reduce dimensionality of input array using PCA.
    
    Args:
        array: Input array to reduce
        n_components: Number of components to reduce to
        
    Returns:
        Fitted PCA model
    """
    pca = PCA(n_components=n_components)
    return pca.fit(array)

def truncate_text(text: str, max_width: float, font_size: int, dpi: int) -> str:
    """
    Truncate text to fit within a specified width.
    
    Args:
        text: Input text to truncate
        max_width: Maximum width in pixels
        font_size: Font size to use
        dpi: Dots per inch for rendering
        
    Returns:
        Truncated text string
    """
    fig, ax = plt.subplots(figsize=(1, 1), dpi=dpi)
    t = ax.text(0, 0, text, fontsize=font_size)
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)
    
    if bbox.width > max_width:
        ratio = max_width / bbox.width
        max_chars = int(len(text) * ratio)
        return f"{text[:max_chars-3]}..."
    return text

def plot_results(df) -> None:
    """
    Plot images with their corresponding text descriptions.
    
    Args:
        df: DataFrame containing image paths and text descriptions
    """
    plt.style.use('seaborn-v0_8')
    
    n_images = len(df)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12*n_rows), dpi=100)
    fig.tight_layout(pad=1.0)
    
    # Handle both 1D and 2D axes arrays
    axes = axes.flatten() if n_rows > 1 else ([axes] if n_cols == 1 else axes)
    
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= len(axes):
            break
            
        img = Image.open(row['image_path'])
        ax = axes[i]
        
        ax.imshow(img)
        ax.axis('off')
        
        # Handle text wrapping and truncation
        img_width = ax.get_window_extent().width
        truncated_text = truncate_text(row['text'], img_width, 8, fig.dpi)
        wrapped_text = textwrap.fill(truncated_text, width=40)
        
        ax.set_title(wrapped_text, fontsize=18, wrap=True, y=1.05)
    
    # Remove unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.show()

def plot_scatter_plot(
    title: str,
    full_data: Dict[str, np.ndarray],
    new_data: Dict[str, np.ndarray]
) -> None:
    """
    Create a scatter plot visualization of embeddings.
    
    Args:
        title: Plot title
        full_data: Dictionary of existing data points
        new_data: Dictionary of new data points to highlight
    """
    plt.style.use('dark_background')
    
    # Setup plot with dark theme
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1C1C1C')
    ax.set_facecolor('#1C1C1C')
    
    # Define color palette
    colors = [
        '#FF6B6B', '#4ECDC4', '#98FB98', '#FFD700', '#FF69B4',
        '#90EE90', '#E6E6FA', '#ADD8E6', '#FF00FF', '#FFA07A'
    ]
    
    # Plot existing data points
    for idx, (key, data) in enumerate(full_data.items()):
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors[idx],
            label=key,
            s=75,
            edgecolors='white'
        )
    
    # Plot new data points with larger markers
    for idx, (key, data) in enumerate(new_data.items(), start=len(full_data)):
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors[idx],
            label=key,
            s=200,
            edgecolors='white'
        )
    
    # Customize plot appearance
    ax.set_xlabel('Feature 1', fontsize=18, color='white')
    ax.set_ylabel('Feature 2', fontsize=18, color='white')
    ax.set_title(title, fontsize=20, color='white', fontweight='bold')
    
    # Customize legend
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        facecolor='#2F2F2F',
        edgecolor='none',
        fontsize=14
    )
    
    # Add grid and customize spines
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def pdf2imgs(
    pdf_path: str,
    pdf_pages_dir: str = "content/Accessibility/pdf_pages"
) -> str:
    """
    Convert PDF pages to individual PNG images.
    
    Args:
        pdf_path: Path to the PDF file
        pdf_pages_dir: Directory to save the generated images
        
    Returns:
        Path to the directory containing the generated images
    """
    import pypdfium2 as pdfium
    
    # Create output directory
    os.makedirs(pdf_pages_dir, exist_ok=True)
    
    # Load PDF document
    pdf = pdfium.PdfDocument(pdf_path)
    
    # Calculate appropriate scale factor
    first_page = pdf.get_page(0)
    resolution = first_page.render().to_numpy().shape
    scale = 1 if max(resolution) >= 1620 else 300 / 72
    
    # Convert each page to PNG
    for page_number in range(len(pdf)):
        page = pdf.get_page(page_number)
        pil_image = page.render(
            scale=scale,
            rotation=0,
            crop=(0, 0, 0, 0),
            may_draw_forms=False,
            fill_color=(255, 255, 255, 255),
            draw_annots=False,
            grayscale=False,
        ).to_pil()
        
        image_path = os.path.join(pdf_pages_dir, f"page_{page_number:03d}.png")
        pil_image.save(image_path)
    
    return pdf_pages_dir

def generate_outputs(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 1000,
    stop_sequences: List[str] = [],
    n: int = 1
) -> List[str]:
    """
    Wrapper function to generate outputs from language models.
    
    Args:
        prompt: Input text prompt
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        stop_sequences: List of sequences where generation should stop
        n: Number of outputs to generate
        
    Returns:
        List of generated outputs
    """
    return run_inference(
        prompt,
        model,
        temperature,
        max_tokens,
        stop_sequences,
        n
    )