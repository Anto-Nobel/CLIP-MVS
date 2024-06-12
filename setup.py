from setuptools import setup, find_packages

setup(
    name="clip_video_processor",
    version="0.1.0",
    description="A library for processing video frames and storing CLIP embeddings in Qdrant",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/clip_video_processor",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0,<=2.0.0",
        "transformers>=4.9.0,<=4.12.0",
        "qdrant-client>=0.9.0,<=1.0.0",
        "opencv-python>=4.5.3.56,<=4.5.5.64",
        "Pillow>=8.3.1,<=9.0.1",
        "matplotlib>=3.4.2,<=3.5.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
