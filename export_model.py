#!/usr/bin/env python3
"""
YOLOv8 Model Converter - PyTorch (.pt) to ONNX
Converts custom YOLOv8 model for mobile Flutter app deployment

Usage:
    python convert_yolo_to_onnx.py

Requirements:
    pip install ultralytics onnx onnxruntime torch torchvision
"""

import os
import sys
from pathlib import Path
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['ultralytics', 'onnx', 'onnxruntime', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def convert_yolo_to_onnx(model_path, output_dir="./converted_models", 
                        input_size=(640, 640), optimize=True):
    """
    Convert YOLOv8 PyTorch model to ONNX format
    
    Args:
        model_path (str): Path to the .pt model file
        output_dir (str): Directory to save converted model
        input_size (tuple): Input image size (height, width)
        optimize (bool): Whether to optimize the model for mobile
    
    Returns:
        str: Path to the converted ONNX model
    """
    
    # Validate input model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not model_path.endswith('.pt'):
        raise ValueError("Input model must be a .pt file")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔄 Loading YOLOv8 model from: {model_path}")
    
    try:
        # Load the custom YOLOv8 model
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {model.names}")
        print(f"   Input size: {input_size}")
        
        # Generate output filename
        model_name = Path(model_path).stem
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        
        print(f"🔄 Converting to ONNX format...")
        
        # Export to ONNX with mobile optimization
        export_result = model.export(
            format='onnx',
            imgsz=input_size,
            optimize=optimize,
            dynamic=False,  # Static input shapes for mobile
            opset=11,       # ONNX opset version (compatible with most mobile frameworks)
            simplify=True,  # Simplify the model graph
        )
        
        # The export method returns the path to the exported file
        exported_path = str(export_result)
        
        # Move to our desired output directory if different
        if exported_path != onnx_path and os.path.exists(exported_path):
            import shutil
            shutil.move(exported_path, onnx_path)
        
        print(f"✅ ONNX model saved to: {onnx_path}")
        
        # Verify the converted model
        verify_onnx_model(onnx_path)
        
        # Generate model info file
        generate_model_info(model, onnx_path, input_size)
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        raise

def verify_onnx_model(onnx_path):
    """Verify that the ONNX model is valid and can be loaded"""
    
    print(f"🔄 Verifying ONNX model...")
    
    try:
        # Load and check the ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model structure is valid")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get model input/output info
        input_info = ort_session.get_inputs()[0]
        output_info = ort_session.get_outputs()
        
        print(f"📊 Model Information:")
        print(f"   Input shape: {input_info.shape}")
        print(f"   Input type: {input_info.type}")
        print(f"   Number of outputs: {len(output_info)}")
        
        for i, output in enumerate(output_info):
            print(f"   Output {i} shape: {output.shape}")
        
        # Get model size
        model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   Model size: {model_size_mb:.2f} MB")
        
        print(f"✅ ONNX model verification completed successfully")
        
    except Exception as e:
        print(f"❌ ONNX model verification failed: {str(e)}")
        raise

def generate_model_info(model, onnx_path, input_size):
    """Generate a JSON info file with model details for Flutter app"""
    
    import json
    
    model_info = {
        "model_name": Path(onnx_path).stem,
        "model_type": "YOLOv8",
        "format": "ONNX",
        "input_size": {
            "width": input_size[1],
            "height": input_size[0]
        },
        "classes": dict(model.names),
        "num_classes": len(model.names),
        "model_path": os.path.basename(onnx_path),
        "preprocessing": {
            "normalize": True,
            "mean": [0.0, 0.0, 0.0],
            "std": [255.0, 255.0, 255.0],
            "format": "RGB"
        },
        "postprocessing": {
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 100
        }
    }
    
    info_path = onnx_path.replace('.onnx', '_info.json')
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"📄 Model info saved to: {info_path}")

def main():
    """Main conversion function"""
    
    print("🚀 YOLOv8 to ONNX Converter for Flutter Mobile App")
    print("=" * 55)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Configuration
    model_path = "yolov8n-custom-v001.pt"  # Your model file
    output_dir = "./assets/models"          # Flutter assets directory
    input_size = (640, 640)                # Standard YOLOv8 input size
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"Please ensure your model file is in the current directory")
        
        # Show files in current directory
        print(f"\nFiles in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.pt'):
                print(f"  📄 {file}")
        
        sys.exit(1)
    
    try:
        # Convert the model
        onnx_path = convert_yolo_to_onnx(
            model_path=model_path,
            output_dir=output_dir,
            input_size=input_size,
            optimize=True
        )
        
        print("\n" + "=" * 55)
        print("🎉 Conversion completed successfully!")
        print("=" * 55)
        print(f"📁 Converted model location: {onnx_path}")
        print(f"📱 Ready for Flutter integration")
        
        print(f"\n📋 Next steps for Flutter integration:")
        print(f"1. Copy the model file to your Flutter project:")
        print(f"   cp {onnx_path} ~/AndroidStudioProjects/tree_detector_mvp/assets/models/")
        print(f"2. Add to pubspec.yaml assets section:")
        print(f"   assets:")
        print(f"     - assets/models/")
        print(f"3. Install ONNX Runtime Flutter package:")
        print(f"   flutter pub add onnxruntime")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()