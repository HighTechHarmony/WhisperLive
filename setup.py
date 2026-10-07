import pathlib
from setuptools import find_packages, setup
from whisper_live.__version__ import __version__


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="whisper_live",
    version=__version__,
    description="A nearly-live implementation of OpenAI's Whisper.",
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/collabora/WhisperLive",
    author="Collabora Ltd",
    author_email="vineet.suryan@collabora.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(
        exclude=(
            "examples",
            "Audio-Transcription-Chrome",
            "Audio-Transcription-Firefox",
            "requirements",
            "whisper-finetuning"
        )
    ),
    package_data={"whisper_live.summarizer_templates": ["SUMMARIZER_TEMPLATE-*.md"]},
    install_requires=[
        "PyAudio",
        "av",
        "faster-whisper==1.2.0",
        "torch==2.11.0+cu128; sys_platform == 'linux' and python_version >= '3.10'",
        "torchaudio==2.11.0+cu128; sys_platform == 'linux' and python_version >= '3.10'",
        "torch; sys_platform != 'linux' or python_version < '3.10'",
        "torchaudio; sys_platform != 'linux' or python_version < '3.10'",
        "websockets",
        "onnxruntime>=1.17.0,<1.20.0; python_version < '3.10'",
        "onnxruntime>=1.20.0,<2; python_version >= '3.10'",
        "scipy",
        "websocket-client",
        "numba",
        "openai-whisper==20250625",
        "kaldialign",
        "soundfile",
        "tokenizers==0.20.3",
        "librosa",
        "numpy>=1.26.4,<2.5",
        "openvino",
        "openvino-genai",
        "openvino-tokenizers",
        "optimum", 
        "optimum-intel",
        "fastapi",
        "uvicorn",
        "python-multipart",
        # CTranslate2 (faster-whisper's backend) uses CUDA 12/cuDNN 9 but
        # doesn't declare the matching wheels as runtime dependencies.
        # PyTorch's default PyPI wheel is CUDA 13; the cu128 build above keeps
        # its NCCL library compatible with CTranslate2's CUDA 12 libraries.
        # Without these CUDA 12 wheels GPU inference dies at first transcription:
        #   ERROR: Library libcublas.so.12 is not found or cannot be loaded
        # Keep them installed so this environment can switch between CPU and GPU.
        "nvidia-cublas-cu12; sys_platform == 'linux'",
        "nvidia-cudnn-cu12; sys_platform == 'linux'",
    ],
    python_requires=">=3.9"
)
