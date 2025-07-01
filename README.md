# 2048 AI Solver with CNN & Reinforcement Learning

[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)](https://tensorflow.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)](https://typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org/)

A sophisticated AI-powered 2048 game solver that combines deep learning, reinforcement learning, and modern web technologies. The AI agent learns to play 2048 through self-play using a custom CNN architecture, with the trained model running entirely in the browser via TensorFlow.js.

## 🎯 Project Overview

This project demonstrates the intersection of machine learning and web development, featuring:

- **🧠 Deep Learning**: Custom CNN model with 2.6M parameters trained via reinforcement learning
- **🎮 Self-Play Training**: AI learns optimal strategies through thousands of game episodes  
- **🌐 Browser AI**: Trained model converted to TensorFlow.js for real-time browser execution
- **⚡ Modern Frontend**: Next.js with TypeScript, real-time AI visualization, and responsive design
- **🚀 Production Ready**: Optimized training pipeline with memory management and GPU acceleration

## 🏗️ Architecture

### Machine Learning Pipeline (Python)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Game Engine   │ -> │   CNN Training   │ -> │  Model Export   │
│   (Python)      │    │ (Reinforcement)  │    │ (TensorFlow.js) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Frontend Application (TypeScript/Next.js)
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Game Display   │ <- │   AI Controller  │ <- │  Trained Model  │
│   (React)       │    │  (TensorFlow.js) │    │   (Browser)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.8+ with pip
- **Git**

### 1. Clone & Setup
```bash
git clone <repository-url>
cd 2048-cnn-solver

# Setup frontend
cd frontend
npm install
cd ..

# Setup training environment
cd training
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### 2. Train the AI Model
```bash
cd training
source venv/bin/activate

# Quick test (10 episodes, ~1 minute)
python train_cnn.py --episodes=10

# Full training (5000 episodes, ~4 hours on CPU, ~1 hour on GPU)
python train_cnn.py --episodes=5000 --batch-size=128 --memory-size=50000
```

### 3. Convert Model for Browser
```bash
# Convert trained model to TensorFlow.js format
python convert_to_tfjs.py --model="models/2048_cnn_model_final.keras" --output="../frontend/public/model"
```

### 4. Run the Web Application
```bash
cd ../frontend
npm run dev
# Open http://localhost:3000
```

## 🧠 AI Training Details

### Model Architecture
- **Input**: 4×4×16 encoded game board (one-hot encoding per tile)
- **CNN Layers**: 3 convolutional layers with ReLU activation
- **Output**: Dual-head architecture (value estimation + policy prediction)
- **Parameters**: ~2.6 million trainable parameters
- **Optimization**: Adam optimizer with mixed precision training

### Training Process
1. **Self-Play Episodes**: AI plays thousands of games against itself
2. **Experience Replay**: Stores game states, actions, and rewards in memory buffer
3. **Q-Learning**: Updates model using temporal difference learning
4. **Epsilon-Greedy**: Balances exploration vs exploitation during training
5. **Target Network**: Stabilizes training with periodic weight updates

### Performance Metrics
- **Training Time**: 4-6 hours for 5000 episodes (CPU), 1-2 hours (GPU)
- **Memory Usage**: Optimized with automatic cleanup, ~500MB peak
- **Game Performance**: Consistently reaches 512+ tiles, often 1024+
- **Browser Speed**: 50-100ms per move prediction

## 🎮 Frontend Features

### Game Interface
- **Interactive 2048 Board**: Smooth animations and responsive design
- **AI Controls**: Play/pause, speed adjustment, step-by-step mode
- **Real-time Statistics**: Score tracking, move analysis, confidence metrics
- **AI Visualization**: Heatmap showing model's tile value predictions

### Technical Implementation
- **TensorFlow.js Integration**: Model loads and runs entirely in browser
- **Real-time Inference**: Sub-100ms move predictions
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Performance Monitoring**: FPS tracking and prediction timing

## 📂 Project Structure

```
2048-cnn-solver/
├── frontend/                 # Next.js web application
│   ├── src/
│   │   ├── components/      # React components (Game, AI controls, etc.)
│   │   ├── lib/            # Game engine and AI model integration
│   │   └── app/            # Next.js app router pages
│   ├── public/             # Static assets
│   └── package.json        # Frontend dependencies
│
├── training/                # Python training pipeline
│   ├── train_cnn.py        # Main training script
│   ├── model.py           # CNN architecture definition
│   ├── game_engine.py     # 2048 game logic
│   ├── trainer.py         # Training utilities
│   ├── convert_to_tfjs.py # Model conversion script
│   ├── test_*.py          # Comprehensive test suite
│   ├── requirements.txt   # Python dependencies
│   └── README.md          # Training documentation
│
├── colab.md               # Google Colab training guide
├── prd.md                # Product requirements document
└── README.md             # This file
```

## 🛠️ Development Workflow

### Training Workflow
1. **Environment Setup**: Create Python virtual environment
2. **Model Training**: Run training script with desired parameters
3. **Performance Testing**: Use memory and integration tests
4. **Model Conversion**: Export to TensorFlow.js format
5. **Browser Integration**: Copy model files to frontend

### Frontend Development
1. **Local Development**: `npm run dev` for hot reload
2. **Model Integration**: Place converted models in `public/model/`
3. **Component Testing**: Interactive testing with live AI
4. **Production Build**: `npm run build` for optimized deployment

## 📊 Training Options

### Local Training (Recommended)
- **Advantages**: No quota limits, runs overnight, your hardware
- **Performance**: ~4 hours for 5000 episodes on modern CPU
- **Command**: `python train_cnn.py --episodes=5000`

### Google Colab Training
- **Advantages**: Free GPU access, faster training (~1 hour)
- **Limitations**: Runtime limits, quota restrictions
- **Guide**: See `colab.md` for detailed setup instructions

### Cloud Alternatives
- **Kaggle Notebooks**: 30 hours/week GPU time
- **Lightning AI**: 22 hours/month free GPU
- **Local GPU**: NVIDIA GPU with CUDA support

## 🧪 Testing

The project includes comprehensive tests:

```bash
cd training
source venv/bin/activate

# Run all tests
python test_training.py      # Training pipeline validation
python test_memory.py        # Memory leak detection  
python test_integration.py  # End-to-end integration
python test_game_engine.py  # Game logic verification
```

## 🚀 Deployment

### Frontend Deployment
The frontend can be deployed to any platform supporting Next.js:
- **Vercel** (recommended): Zero-config deployment
- **Netlify**: Static site generation support
- **AWS/GCP/Azure**: Full server deployment options

### Model Serving
- **Client-Side**: TensorFlow.js runs entirely in browser
- **Edge Computing**: Deploy to CDN for global distribution
- **No Backend Required**: Complete frontend-only architecture

## 🔧 Configuration

### Training Parameters
```bash
# Key training options
--episodes=5000           # Number of training episodes
--batch-size=128         # Training batch size
--memory-size=50000      # Experience replay buffer size
--learning-rate=0.001    # Adam optimizer learning rate
--epsilon-decay=300000   # Exploration decay schedule
--save-interval=500      # Model checkpoint frequency
```

### Model Architecture
- Configurable in `training/model.py`
- Supports different CNN depths and filter sizes
- Adjustable output heads for different game variants

## 📈 Performance Benchmarks

| Metric | Local CPU | Google Colab GPU |
|--------|-----------|------------------|
| Training Time (1000 episodes) | ~50 minutes | ~12 minutes |
| Training Time (5000 episodes) | ~4 hours | ~1 hour |
| Memory Usage | ~500MB peak | ~2GB peak |
| Model Size | 50MB (.keras) | 15MB (.tfjs) |
| Browser Inference | 50-100ms | 50-100ms |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run the test suite: `python test_*.py`
5. Commit changes: `git commit -m "Description"`
6. Push to branch: `git push origin feature-name`
7. Open a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TensorFlow Team**: For the incredible ML framework
- **Next.js Team**: For the outstanding React framework  
- **2048 Game**: Original concept by Gabriele Cirulli
- **OpenAI**: For research inspiration on game-playing AI

## 📞 Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See `training/README.md` for detailed training docs
- **Colab Guide**: Check `colab.md` for cloud training setup

---

Built with ❤️ by passionate developers exploring the intersection of AI and web technology. 