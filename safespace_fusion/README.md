# SafeSpace Fusion - Multi-Modal Stress Analysis System

A comprehensive stress analysis system that combines multiple data modalities to provide accurate stress level assessments.

## Features

- **Multi-Modal Analysis**: Combines biomedical, behavioral, text, voice, and facial data
- **Real-time Processing**: Advanced fusion engine with residual-persistence analysis
- **User-Friendly Interface**: Modern web-based frontend for easy data input
- **Comprehensive Results**: Detailed breakdown of stress levels and modality contributions

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure all model files are in their correct locations as per the project structure.

## Usage

### Web Interface

1. Start the Flask application:
```bash
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
├── models/               # ML models and data
├── templates/            # HTML templates
├── uploads/              # User uploaded files
└── requirements.txt      # Python dependencies
```

## Contributing

Feel free to submit issues and enhancement requests! 