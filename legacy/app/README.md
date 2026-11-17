# SafeSpace Fusion - Multi-Modal Stress Analysis System

A comprehensive stress analysis system that combines multiple data modalities to provide accurate stress level assessments.

## Features

- **Multi-Modal Analysis**: Combines biomedical, behavioral, text, voice, and facial data
- **Real-time Processing**: Advanced fusion engine with residual-persistence analysis
- **User-Friendly Interface**: Modern web-based frontend for easy data input
- **Comprehensive Results**: Detailed breakdown of stress levels and modality contributions

## Installation

From the repository root:

1. Create and activate a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the required dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r safespace_fusion/requirements.txt
```

3. Download the model artifacts:

```bash
python scripts/download_models.py --repo-id parth18s/safespace-fusion-models
python scripts/check_models.py
```

For a private Hugging Face model repo, log in before downloading:

```bash
hf auth login
```

## Uploading Model Artifacts

Normal GitHub repositories reject files larger than 100 MB, and this project has
several model files above that limit. The largest local file,
`safespace_fusion/models/elm_weights-001.npz`, is about 2.3 GB, so it may also
exceed GitHub Free/Pro Git LFS limits.

Recommended workflow:

1. Push the code to GitHub.
2. Upload model artifacts to a Hugging Face model repo:

```bash
hf auth login
python scripts/upload_models_to_hf.py --repo-id parth18s/safespace-fusion-models
```

By default the helper creates a private Hugging Face repo. Add `--public` only
if the model artifacts are safe to share publicly:

```bash
python scripts/upload_models_to_hf.py --repo-id parth18s/safespace-fusion-models --public
```

3. Send teammates the GitHub repo URL and the Hugging Face repo id.

Do not commit local virtual environments such as `safespace/` or `.venv/`.
Teammates should recreate their own environment from `requirements.txt`.

## Usage

### Web Interface

1. Start the Flask application:

```bash
cd safespace_fusion
python app.py
```

2. Open your web browser and navigate to `http://localhost:8080`

3. Fill in the form with your data:
   - **Text Analysis**: Describe your current emotional state
   - **Biomedical Data**: Enter sensor readings (WAX, WAY, WAZ, BVP, EDA, Temperature)
   - **Behavioral Data**: Input sleep, activity, and lifestyle information
   - **File Uploads**: Upload voice recordings and facial images (optional)

4. Click "Analyze Stress Level" to get your results

### Command Line Interface

You can also run the original command-line version:

```bash
cd safespace_fusion
python main.py
```

## Input Parameters

### Biomedical Features
- **WAX**: Accelerometer X-axis reading (-35 to -30)
- **WAY**: Accelerometer Y-axis reading (-60 to -50)
- **WAZ**: Accelerometer Z-axis reading (3 to 6)
- **BVP**: Blood Volume Pulse (-0.9 to -0.7)
- **EDA**: Electrodermal Activity (0.3 to 0.5)
- **Temperature**: Body temperature in Celsius (31 to 33)

### Behavioral Data
- **Sleep Duration**: Hours of sleep (0-24)
- **Sleep Quality**: Subjective sleep quality rating (1-10)
- **Physical Activity**: Minutes of physical activity per day
- **Daily Steps**: Number of steps taken
- **BMI Category**: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese

### File Uploads
- **Voice Recording**: WAV or MP3 files for voice stress analysis
- **Facial Image**: JPG or PNG images for facial stress analysis

## Output

The system provides:
- **WMSS Score**: Weighted Multi-Modal Stress Score (0-1)
- **Stress Level**: Low, Medium, or High stress classification
- **Dominant Modality**: Which input type contributed most to the stress assessment
- **Individual Scores**: Stress scores from each modality
- **Fusion Details**: Internal EMA, residual, and persistence values

## Technical Details

The system uses a sophisticated fusion engine that:
- Applies Exponential Moving Average (EMA) smoothing
- Calculates residual values for change detection
- Tracks persistence for stability assessment
- Combines modalities using weighted fusion

## Project Structure

```
safespace_fusion/
├── app.py                 # Flask web application
├── main.py               # Original command-line interface
├── fusion_engine.py      # Multi-modal fusion algorithm
├── models/               # Model loaders plus downloaded artifacts
├── templates/            # HTML templates
├── uploads/              # User uploaded files
└── requirements.txt      # Python dependencies
```

## Contributing

Feel free to submit issues and enhancement requests! 
