/**
 * Curriculum registry for Learn Deep Learning.
 * 18 subjects covering deep learning from basics to research frontiers.
 */

export const CURRICULUM = [
  {
    id: '01-foundations',
    title: 'Neural Network Foundations',
    icon: '∇',
    colorHex: '#6366f1',
    description: 'The building blocks of deep learning — perceptrons, activation functions, loss functions, and the universal approximation theorem.',
    prerequisites: [],
    dlRelevance: 95,
    estimatedHours: 35,
    difficulty: 'beginner',
    chapters: [
      {
        id: 'c1-perceptron',
        title: 'The Perceptron',
        description: 'The simplest neural network — a single neuron that learns a linear decision boundary.',
        difficulty: 'beginner',
        estimatedMinutes: 240,
        sections: [
          { id: 's1-biological-neuron', title: 'From Biology to Math', difficulty: 'beginner', readingMinutes: 20, description: 'How biological neurons inspired the perceptron model.', buildsOn: null },
          { id: 's2-perceptron-algorithm', title: 'Perceptron Algorithm', difficulty: 'beginner', readingMinutes: 25, description: 'The perceptron learning rule and convergence theorem.', buildsOn: '01-foundations/c1-perceptron/s1-biological-neuron' },
          { id: 's3-linear-separability', title: 'Linear Separability & Limits', difficulty: 'beginner', readingMinutes: 20, description: 'XOR problem and why single perceptrons are limited.', buildsOn: '01-foundations/c1-perceptron/s2-perceptron-algorithm' },
        ],
      },
      {
        id: 'c2-activation-functions',
        title: 'Activation Functions',
        description: 'Non-linear transformations that give neural networks their power.',
        difficulty: 'beginner',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-sigmoid-tanh', title: 'Sigmoid & Tanh', difficulty: 'beginner', readingMinutes: 20, description: 'Classic smooth activations and their properties.', buildsOn: '01-foundations/c1-perceptron/s3-linear-separability' },
          { id: 's2-relu-family', title: 'ReLU & Variants', difficulty: 'beginner', readingMinutes: 22, description: 'ReLU, Leaky ReLU, PReLU, ELU — modern piecewise activations.', buildsOn: '01-foundations/c2-activation-functions/s1-sigmoid-tanh' },
          { id: 's3-modern-activations', title: 'GELU, Swish & SiLU', difficulty: 'intermediate', readingMinutes: 20, description: 'Smooth approximations used in Transformers and modern architectures.', buildsOn: '01-foundations/c2-activation-functions/s2-relu-family' },
        ],
      },
      {
        id: 'c3-loss-functions',
        title: 'Loss Functions & Objectives',
        description: 'How we measure the gap between predictions and truth — the signal that drives learning.',
        difficulty: 'beginner',
        estimatedMinutes: 300,
        sections: [
          { id: 's1-regression-losses', title: 'Regression Losses', difficulty: 'beginner', readingMinutes: 20, description: 'MSE, MAE, Huber loss and their gradients.', buildsOn: '01-foundations/c2-activation-functions/s3-modern-activations' },
          { id: 's2-classification-losses', title: 'Classification Losses', difficulty: 'beginner', readingMinutes: 25, description: 'Cross-entropy, softmax, focal loss.', buildsOn: '01-foundations/c3-loss-functions/s1-regression-losses' },
          { id: 's3-advanced-losses', title: 'Contrastive & Triplet Losses', difficulty: 'intermediate', readingMinutes: 25, description: 'Metric learning losses for embeddings and similarity.', buildsOn: '01-foundations/c3-loss-functions/s2-classification-losses' },
        ],
      },
      {
        id: 'c4-universal-approximation',
        title: 'Universal Approximation',
        description: 'The theoretical foundation — why neural networks can learn any function.',
        difficulty: 'intermediate',
        estimatedMinutes: 250,
        sections: [
          { id: 's1-uat', title: 'Universal Approximation Theorem', difficulty: 'intermediate', readingMinutes: 25, description: 'Cybenko and Hornik theorems for width and depth.', buildsOn: '01-foundations/c3-loss-functions/s3-advanced-losses' },
          { id: 's2-depth-vs-width', title: 'Depth vs Width', difficulty: 'intermediate', readingMinutes: 22, description: 'Why deep networks are more efficient than wide ones.', buildsOn: '01-foundations/c4-universal-approximation/s1-uat' },
          { id: 's3-expressiveness', title: 'Expressiveness & Complexity', difficulty: 'intermediate', readingMinutes: 25, description: 'VC dimension, Rademacher complexity, and what networks can represent.', buildsOn: '01-foundations/c4-universal-approximation/s2-depth-vs-width' },
        ],
      },
    ],
  },
  {
    id: '02-backpropagation',
    title: 'Backpropagation & Training',
    icon: '⟲',
    colorHex: '#8b5cf6',
    description: 'The engine of deep learning — computational graphs, the chain rule, automatic differentiation, and gradient-based optimization.',
    prerequisites: ['01-foundations'],
    dlRelevance: 98,
    estimatedHours: 40,
    difficulty: 'beginner',
    chapters: [
      {
        id: 'c1-computational-graphs',
        title: 'Computational Graphs',
        description: 'Representing neural networks as directed acyclic graphs of operations.',
        difficulty: 'beginner',
        estimatedMinutes: 240,
        sections: [
          { id: 's1-forward-pass', title: 'Forward Pass', difficulty: 'beginner', readingMinutes: 20, description: 'Computing outputs layer by layer through the graph.', buildsOn: '01-foundations/c4-universal-approximation/s3-expressiveness' },
          { id: 's2-backward-pass', title: 'Backward Pass', difficulty: 'beginner', readingMinutes: 25, description: 'Propagating gradients backward through the graph.', buildsOn: '02-backpropagation/c1-computational-graphs/s1-forward-pass' },
          { id: 's3-graph-operations', title: 'Graph Operations & Jacobians', difficulty: 'intermediate', readingMinutes: 25, description: 'Local gradients, Jacobian matrices, and vector-Jacobian products.', buildsOn: '02-backpropagation/c1-computational-graphs/s2-backward-pass' },
        ],
      },
      {
        id: 'c2-chain-rule',
        title: 'The Chain Rule & Backprop',
        description: 'How the chain rule of calculus enables efficient gradient computation in deep networks.',
        difficulty: 'beginner',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-chain-rule-review', title: 'Chain Rule in Deep Networks', difficulty: 'beginner', readingMinutes: 22, description: 'Multivariate chain rule applied to layered compositions.', buildsOn: '02-backpropagation/c1-computational-graphs/s3-graph-operations' },
          { id: 's2-backprop-algorithm', title: 'Backpropagation Algorithm', difficulty: 'beginner', readingMinutes: 28, description: 'Step-by-step backprop with worked examples for MLPs.', buildsOn: '02-backpropagation/c2-chain-rule/s1-chain-rule-review' },
          { id: 's3-vanishing-exploding', title: 'Vanishing & Exploding Gradients', difficulty: 'intermediate', readingMinutes: 25, description: 'Why gradients can vanish or explode and how to mitigate it.', buildsOn: '02-backpropagation/c2-chain-rule/s2-backprop-algorithm' },
        ],
      },
      {
        id: 'c3-autodiff',
        title: 'Automatic Differentiation',
        description: 'How PyTorch and TensorFlow compute gradients automatically.',
        difficulty: 'intermediate',
        estimatedMinutes: 260,
        sections: [
          { id: 's1-forward-mode', title: 'Forward-Mode AD', difficulty: 'intermediate', readingMinutes: 22, description: 'Dual numbers and forward accumulation.', buildsOn: '02-backpropagation/c2-chain-rule/s3-vanishing-exploding' },
          { id: 's2-reverse-mode', title: 'Reverse-Mode AD', difficulty: 'intermediate', readingMinutes: 25, description: 'Tape-based reverse accumulation — how PyTorch autograd works.', buildsOn: '02-backpropagation/c3-autodiff/s1-forward-mode' },
          { id: 's3-higher-order', title: 'Higher-Order Gradients', difficulty: 'advanced', readingMinutes: 22, description: 'Hessians, Hessian-vector products, and second-order methods.', buildsOn: '02-backpropagation/c3-autodiff/s2-reverse-mode' },
        ],
      },
      {
        id: 'c4-optimization',
        title: 'Gradient-Based Optimization',
        description: 'From vanilla gradient descent to the fundamentals of training neural networks.',
        difficulty: 'beginner',
        estimatedMinutes: 300,
        sections: [
          { id: 's1-gradient-descent', title: 'Gradient Descent', difficulty: 'beginner', readingMinutes: 22, description: 'Batch, mini-batch, and stochastic gradient descent.', buildsOn: '02-backpropagation/c3-autodiff/s3-higher-order' },
          { id: 's2-loss-landscapes', title: 'Loss Landscapes', difficulty: 'intermediate', readingMinutes: 25, description: 'Saddle points, local minima, and the geometry of optimization.', buildsOn: '02-backpropagation/c4-optimization/s1-gradient-descent' },
          { id: 's3-training-loop', title: 'The Training Loop', difficulty: 'beginner', readingMinutes: 25, description: 'Epochs, batches, validation, and practical training in PyTorch.', buildsOn: '02-backpropagation/c4-optimization/s2-loss-landscapes' },
        ],
      },
    ],
  },
  {
    id: '03-regularization',
    title: 'Regularization & Generalization',
    icon: '◈',
    colorHex: '#a855f7',
    description: 'Techniques to prevent overfitting and ensure models generalize — from weight decay to data augmentation.',
    prerequisites: ['02-backpropagation'],
    dlRelevance: 90,
    estimatedHours: 30,
    difficulty: 'intermediate',
    chapters: [
      {
        id: 'c1-overfitting',
        title: 'Overfitting & Bias-Variance',
        description: 'Understanding when and why models fail to generalize.',
        difficulty: 'beginner',
        estimatedMinutes: 200,
        sections: [
          { id: 's1-bias-variance', title: 'Bias-Variance Tradeoff', difficulty: 'beginner', readingMinutes: 20, description: 'Decomposing error into bias, variance, and irreducible noise.', buildsOn: '02-backpropagation/c4-optimization/s3-training-loop' },
          { id: 's2-overfitting-detection', title: 'Detecting Overfitting', difficulty: 'beginner', readingMinutes: 18, description: 'Train/val curves, gap analysis, and diagnostic techniques.', buildsOn: '03-regularization/c1-overfitting/s1-bias-variance' },
          { id: 's3-double-descent', title: 'Double Descent', difficulty: 'advanced', readingMinutes: 22, description: 'The surprising phenomenon where more parameters help generalization.', buildsOn: '03-regularization/c1-overfitting/s2-overfitting-detection' },
        ],
      },
      {
        id: 'c2-weight-regularization',
        title: 'Weight Regularization',
        description: 'Constraining model complexity through weight penalties.',
        difficulty: 'intermediate',
        estimatedMinutes: 220,
        sections: [
          { id: 's1-l1-l2', title: 'L1 & L2 Regularization', difficulty: 'intermediate', readingMinutes: 22, description: 'Weight decay, sparsity, and Bayesian interpretation.', buildsOn: '03-regularization/c1-overfitting/s3-double-descent' },
          { id: 's2-weight-decay', title: 'Decoupled Weight Decay', difficulty: 'intermediate', readingMinutes: 20, description: 'AdamW and why decoupled weight decay matters.', buildsOn: '03-regularization/c2-weight-regularization/s1-l1-l2' },
          { id: 's3-spectral-norm', title: 'Spectral Normalization', difficulty: 'advanced', readingMinutes: 22, description: 'Constraining the Lipschitz constant of layers.', buildsOn: '03-regularization/c2-weight-regularization/s2-weight-decay' },
        ],
      },
      {
        id: 'c3-dropout',
        title: 'Dropout & Noise',
        description: 'Stochastic regularization through random deactivation.',
        difficulty: 'intermediate',
        estimatedMinutes: 200,
        sections: [
          { id: 's1-dropout', title: 'Dropout', difficulty: 'intermediate', readingMinutes: 20, description: 'Random neuron deactivation as implicit ensemble.', buildsOn: '03-regularization/c2-weight-regularization/s3-spectral-norm' },
          { id: 's2-dropconnect', title: 'DropConnect & DropPath', difficulty: 'intermediate', readingMinutes: 18, description: 'Dropping connections and paths instead of neurons.', buildsOn: '03-regularization/c3-dropout/s1-dropout' },
          { id: 's3-label-smoothing', title: 'Label Smoothing & Mixup', difficulty: 'intermediate', readingMinutes: 22, description: 'Soft targets and interpolation-based regularization.', buildsOn: '03-regularization/c3-dropout/s2-dropconnect' },
        ],
      },
      {
        id: 'c4-early-stopping',
        title: 'Early Stopping & Validation',
        description: 'Using validation performance to prevent overfitting.',
        difficulty: 'beginner',
        estimatedMinutes: 160,
        sections: [
          { id: 's1-early-stopping', title: 'Early Stopping', difficulty: 'beginner', readingMinutes: 18, description: 'Patience, checkpointing, and the role of validation.', buildsOn: '03-regularization/c3-dropout/s3-label-smoothing' },
          { id: 's2-cross-validation', title: 'Cross-Validation for DL', difficulty: 'intermediate', readingMinutes: 20, description: 'K-fold and its challenges for large models.', buildsOn: '03-regularization/c4-early-stopping/s1-early-stopping' },
          { id: 's3-hyperparameter-tuning', title: 'Hyperparameter Tuning', difficulty: 'intermediate', readingMinutes: 22, description: 'Grid search, random search, and Bayesian optimization.', buildsOn: '03-regularization/c4-early-stopping/s2-cross-validation' },
        ],
      },
      {
        id: 'c5-data-augmentation',
        title: 'Data Augmentation',
        description: 'Creating training diversity through input transformations.',
        difficulty: 'intermediate',
        estimatedMinutes: 220,
        sections: [
          { id: 's1-image-augmentation', title: 'Image Augmentation', difficulty: 'beginner', readingMinutes: 20, description: 'Flips, crops, color jitter, and random erasing.', buildsOn: '03-regularization/c4-early-stopping/s3-hyperparameter-tuning' },
          { id: 's2-advanced-augmentation', title: 'AutoAugment & RandAugment', difficulty: 'intermediate', readingMinutes: 22, description: 'Learned and randomized augmentation policies.', buildsOn: '03-regularization/c5-data-augmentation/s1-image-augmentation' },
          { id: 's3-text-audio-augmentation', title: 'Text & Audio Augmentation', difficulty: 'intermediate', readingMinutes: 20, description: 'Back-translation, synonym replacement, SpecAugment.', buildsOn: '03-regularization/c5-data-augmentation/s2-advanced-augmentation' },
        ],
      },
    ],
  },
  {
    id: '04-optimization',
    title: 'Advanced Optimization',
    icon: '⤵',
    colorHex: '#ec4899',
    description: 'Sophisticated optimization algorithms, learning rate strategies, normalization, and initialization techniques for training deep networks.',
    prerequisites: ['02-backpropagation'],
    dlRelevance: 95,
    estimatedHours: 35,
    difficulty: 'intermediate',
    chapters: [
      {
        id: 'c1-sgd-variants',
        title: 'SGD & Momentum',
        description: 'Classical momentum methods for accelerating gradient descent.',
        difficulty: 'intermediate',
        estimatedMinutes: 220,
        sections: [
          { id: 's1-momentum', title: 'Momentum', difficulty: 'intermediate', readingMinutes: 22, description: 'Classical and Nesterov momentum for faster convergence.', buildsOn: '02-backpropagation/c4-optimization/s3-training-loop' },
          { id: 's2-nesterov', title: 'Nesterov Accelerated Gradient', difficulty: 'intermediate', readingMinutes: 22, description: 'Look-ahead gradient for better convergence.', buildsOn: '04-optimization/c1-sgd-variants/s1-momentum' },
          { id: 's3-convergence', title: 'Convergence Analysis', difficulty: 'advanced', readingMinutes: 25, description: 'Convergence rates and theoretical guarantees.', buildsOn: '04-optimization/c1-sgd-variants/s2-nesterov' },
        ],
      },
      {
        id: 'c2-adaptive-methods',
        title: 'Adam, AdaGrad, RMSProp',
        description: 'Adaptive learning rate methods that adjust per-parameter.',
        difficulty: 'intermediate',
        estimatedMinutes: 260,
        sections: [
          { id: 's1-adagrad', title: 'AdaGrad & RMSProp', difficulty: 'intermediate', readingMinutes: 22, description: 'Per-parameter learning rates from gradient history.', buildsOn: '04-optimization/c1-sgd-variants/s3-convergence' },
          { id: 's2-adam', title: 'Adam & AdamW', difficulty: 'intermediate', readingMinutes: 25, description: 'Combining momentum with adaptive rates — the default optimizer.', buildsOn: '04-optimization/c2-adaptive-methods/s1-adagrad' },
          { id: 's3-lion-sophia', title: 'Lion, Sophia & Modern Optimizers', difficulty: 'advanced', readingMinutes: 25, description: 'Sign-based and second-order modern optimizers.', buildsOn: '04-optimization/c2-adaptive-methods/s2-adam' },
        ],
      },
      {
        id: 'c3-learning-rate',
        title: 'Learning Rate Scheduling',
        description: 'Dynamic adjustment of learning rates during training.',
        difficulty: 'intermediate',
        estimatedMinutes: 220,
        sections: [
          { id: 's1-warmup-decay', title: 'Warmup & Decay', difficulty: 'intermediate', readingMinutes: 20, description: 'Linear warmup, step decay, exponential decay.', buildsOn: '04-optimization/c2-adaptive-methods/s3-lion-sophia' },
          { id: 's2-cosine-annealing', title: 'Cosine Annealing', difficulty: 'intermediate', readingMinutes: 20, description: 'Cosine schedules and warm restarts.', buildsOn: '04-optimization/c3-learning-rate/s1-warmup-decay' },
          { id: 's3-cyclic-lr', title: 'Cyclic & One-Cycle Policies', difficulty: 'intermediate', readingMinutes: 22, description: 'Super-convergence through cyclic learning rates.', buildsOn: '04-optimization/c3-learning-rate/s2-cosine-annealing' },
        ],
      },
      {
        id: 'c4-batch-normalization',
        title: 'Batch & Layer Normalization',
        description: 'Normalizing activations for faster and more stable training.',
        difficulty: 'intermediate',
        estimatedMinutes: 260,
        sections: [
          { id: 's1-batch-norm', title: 'Batch Normalization', difficulty: 'intermediate', readingMinutes: 25, description: 'Normalizing mini-batch statistics for internal covariate shift.', buildsOn: '04-optimization/c3-learning-rate/s3-cyclic-lr' },
          { id: 's2-layer-norm', title: 'Layer Normalization', difficulty: 'intermediate', readingMinutes: 22, description: 'Instance-level normalization used in Transformers.', buildsOn: '04-optimization/c4-batch-normalization/s1-batch-norm' },
          { id: 's3-rmsnorm-group', title: 'RMSNorm & Group Norm', difficulty: 'intermediate', readingMinutes: 22, description: 'Simplified norms and group-based alternatives.', buildsOn: '04-optimization/c4-batch-normalization/s2-layer-norm' },
        ],
      },
      {
        id: 'c5-initialization',
        title: 'Weight Initialization',
        description: 'Starting right — initialization strategies for stable training.',
        difficulty: 'intermediate',
        estimatedMinutes: 200,
        sections: [
          { id: 's1-xavier-he', title: 'Xavier & He Initialization', difficulty: 'intermediate', readingMinutes: 22, description: 'Variance-preserving initialization for sigmoid and ReLU networks.', buildsOn: '04-optimization/c4-batch-normalization/s3-rmsnorm-group' },
          { id: 's2-orthogonal', title: 'Orthogonal & LSUV', difficulty: 'intermediate', readingMinutes: 20, description: 'Orthogonal matrices and data-dependent initialization.', buildsOn: '04-optimization/c5-initialization/s1-xavier-he' },
          { id: 's3-fixup-rezero', title: 'Fixup & ReZero', difficulty: 'advanced', readingMinutes: 22, description: 'Initialization techniques for very deep residual networks.', buildsOn: '04-optimization/c5-initialization/s2-orthogonal' },
        ],
      },
    ],
  },
  {
    id: '05-cnns',
    title: 'Convolutional Neural Networks',
    icon: '▦',
    colorHex: '#f97316',
    description: 'Spatial feature learning through convolutions — from LeNet to modern architectures for images and beyond.',
    prerequisites: ['03-regularization'],
    dlRelevance: 95,
    estimatedHours: 45,
    difficulty: 'intermediate',
    chapters: [
      {
        id: 'c1-convolution-operation',
        title: 'The Convolution Operation',
        description: 'Mathematical foundations of discrete convolution in neural networks.',
        difficulty: 'intermediate',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-discrete-convolution', title: 'Discrete Convolution', difficulty: 'intermediate', readingMinutes: 25, description: '1D, 2D convolution, cross-correlation, and kernel operations.', buildsOn: '03-regularization/c5-data-augmentation/s3-text-audio-augmentation' },
          { id: 's2-stride-dilation', title: 'Stride, Dilation & Transposed', difficulty: 'intermediate', readingMinutes: 22, description: 'Strided, dilated, and transposed convolutions for varied receptive fields.', buildsOn: '05-cnns/c1-convolution-operation/s1-discrete-convolution' },
          { id: 's3-depthwise-separable', title: 'Depthwise Separable Convolutions', difficulty: 'intermediate', readingMinutes: 22, description: 'Efficient convolutions from MobileNet for reduced computation.', buildsOn: '05-cnns/c1-convolution-operation/s2-stride-dilation' },
        ],
      },
      {
        id: 'c2-pooling-padding',
        title: 'Pooling & Padding',
        description: 'Spatial downsampling and boundary handling in CNNs.',
        difficulty: 'beginner',
        estimatedMinutes: 200,
        sections: [
          { id: 's1-pooling', title: 'Max & Average Pooling', difficulty: 'beginner', readingMinutes: 18, description: 'Spatial downsampling and translation invariance.', buildsOn: '05-cnns/c1-convolution-operation/s3-depthwise-separable' },
          { id: 's2-global-pooling', title: 'Global Average Pooling', difficulty: 'intermediate', readingMinutes: 18, description: 'Replacing fully-connected layers with global pooling.', buildsOn: '05-cnns/c2-pooling-padding/s1-pooling' },
          { id: 's3-receptive-field', title: 'Receptive Field Analysis', difficulty: 'intermediate', readingMinutes: 22, description: 'Computing effective receptive fields in deep CNNs.', buildsOn: '05-cnns/c2-pooling-padding/s2-global-pooling' },
        ],
      },
      {
        id: 'c3-classic-architectures',
        title: 'Classic Architectures: LeNet to ResNet',
        description: 'The evolution of CNN architectures that shaped deep learning.',
        difficulty: 'intermediate',
        estimatedMinutes: 350,
        sections: [
          { id: 's1-lenet-alexnet', title: 'LeNet & AlexNet', difficulty: 'beginner', readingMinutes: 22, description: 'The first CNNs — handwriting recognition to ImageNet breakthrough.', buildsOn: '05-cnns/c2-pooling-padding/s3-receptive-field' },
          { id: 's2-vgg-inception', title: 'VGG & GoogLeNet/Inception', difficulty: 'intermediate', readingMinutes: 25, description: 'Deeper networks and multi-scale feature extraction.', buildsOn: '05-cnns/c3-classic-architectures/s1-lenet-alexnet' },
          { id: 's3-resnet', title: 'ResNet & Skip Connections', difficulty: 'intermediate', readingMinutes: 28, description: 'Residual learning and the power of skip connections.', buildsOn: '05-cnns/c3-classic-architectures/s2-vgg-inception' },
        ],
      },
      {
        id: 'c4-modern-architectures',
        title: 'Modern CNNs: EfficientNet, ConvNeXt',
        description: 'State-of-the-art CNN designs with neural architecture search and modern training.',
        difficulty: 'advanced',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-efficientnet', title: 'EfficientNet & Compound Scaling', difficulty: 'advanced', readingMinutes: 25, description: 'Neural architecture search and balanced width/depth/resolution scaling.', buildsOn: '05-cnns/c3-classic-architectures/s3-resnet' },
          { id: 's2-convnext', title: 'ConvNeXt', difficulty: 'advanced', readingMinutes: 25, description: 'Modernizing CNNs with Transformer-era training recipes.', buildsOn: '05-cnns/c4-modern-architectures/s1-efficientnet' },
          { id: 's3-nas', title: 'Neural Architecture Search', difficulty: 'advanced', readingMinutes: 28, description: 'Automated discovery of optimal architectures.', buildsOn: '05-cnns/c4-modern-architectures/s2-convnext' },
        ],
      },
      {
        id: 'c5-object-detection',
        title: 'Object Detection: YOLO, Faster R-CNN',
        description: 'Detecting and localizing objects in images.',
        difficulty: 'intermediate',
        estimatedMinutes: 300,
        sections: [
          { id: 's1-two-stage', title: 'Two-Stage Detectors', difficulty: 'intermediate', readingMinutes: 25, description: 'R-CNN, Fast R-CNN, Faster R-CNN and region proposals.', buildsOn: '05-cnns/c4-modern-architectures/s3-nas' },
          { id: 's2-one-stage', title: 'One-Stage Detectors', difficulty: 'intermediate', readingMinutes: 25, description: 'YOLO, SSD, and anchor-free detection.', buildsOn: '05-cnns/c5-object-detection/s1-two-stage' },
          { id: 's3-detr', title: 'DETR & End-to-End Detection', difficulty: 'advanced', readingMinutes: 25, description: 'Transformer-based detection eliminating hand-crafted components.', buildsOn: '05-cnns/c5-object-detection/s2-one-stage' },
        ],
      },
      {
        id: 'c6-semantic-segmentation',
        title: 'Semantic Segmentation',
        description: 'Pixel-wise classification for scene understanding.',
        difficulty: 'intermediate',
        estimatedMinutes: 260,
        sections: [
          { id: 's1-fcn-unet', title: 'FCN & U-Net', difficulty: 'intermediate', readingMinutes: 25, description: 'Fully convolutional networks and encoder-decoder architectures.', buildsOn: '05-cnns/c5-object-detection/s3-detr' },
          { id: 's2-deeplab', title: 'DeepLab & Atrous Convolutions', difficulty: 'intermediate', readingMinutes: 22, description: 'Dilated convolutions and CRF-based refinement.', buildsOn: '05-cnns/c6-semantic-segmentation/s1-fcn-unet' },
          { id: 's3-panoptic', title: 'Instance & Panoptic Segmentation', difficulty: 'advanced', readingMinutes: 25, description: 'Mask R-CNN and unified panoptic segmentation.', buildsOn: '05-cnns/c6-semantic-segmentation/s2-deeplab' },
        ],
      },
    ],
  },
  {
    id: '06-rnns',
    title: 'Recurrent Neural Networks',
    icon: '∞',
    colorHex: '#14b8a6',
    description: 'Sequential data processing — vanilla RNNs, LSTM, GRU, and sequence-to-sequence models.',
    prerequisites: ['03-regularization'],
    dlRelevance: 85,
    estimatedHours: 35,
    difficulty: 'intermediate',
    chapters: [
      {
        id: 'c1-vanilla-rnn',
        title: 'Vanilla RNN',
        description: 'The simplest recurrent architecture and its fundamental properties.',
        difficulty: 'intermediate',
        estimatedMinutes: 240,
        sections: [
          { id: 's1-rnn-basics', title: 'RNN Architecture', difficulty: 'intermediate', readingMinutes: 22, description: 'Hidden states, recurrent connections, and unrolling through time.', buildsOn: '03-regularization/c5-data-augmentation/s3-text-audio-augmentation' },
          { id: 's2-bptt', title: 'Backpropagation Through Time', difficulty: 'intermediate', readingMinutes: 25, description: 'Gradient computation in unrolled recurrent networks.', buildsOn: '06-rnns/c1-vanilla-rnn/s1-rnn-basics' },
          { id: 's3-rnn-applications', title: 'RNN Applications', difficulty: 'intermediate', readingMinutes: 20, description: 'Sequence classification, generation, and language modeling.', buildsOn: '06-rnns/c1-vanilla-rnn/s2-bptt' },
        ],
      },
      {
        id: 'c2-lstm',
        title: 'LSTM',
        description: 'Long Short-Term Memory — gated cells for learning long-range dependencies.',
        difficulty: 'intermediate',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-lstm-gates', title: 'LSTM Gates & Cell State', difficulty: 'intermediate', readingMinutes: 25, description: 'Forget, input, output gates and the cell state highway.', buildsOn: '06-rnns/c1-vanilla-rnn/s3-rnn-applications' },
          { id: 's2-lstm-variants', title: 'LSTM Variants', difficulty: 'intermediate', readingMinutes: 22, description: 'Peephole connections, coupled gates, and more.', buildsOn: '06-rnns/c2-lstm/s1-lstm-gates' },
          { id: 's3-lstm-training', title: 'Training LSTMs', difficulty: 'intermediate', readingMinutes: 22, description: 'Gradient clipping, initialization, and practical tips.', buildsOn: '06-rnns/c2-lstm/s2-lstm-variants' },
        ],
      },
      {
        id: 'c3-gru',
        title: 'GRU',
        description: 'Gated Recurrent Unit — a simplified alternative to LSTM.',
        difficulty: 'intermediate',
        estimatedMinutes: 180,
        sections: [
          { id: 's1-gru-architecture', title: 'GRU Architecture', difficulty: 'intermediate', readingMinutes: 20, description: 'Reset and update gates in a simplified gated RNN.', buildsOn: '06-rnns/c2-lstm/s3-lstm-training' },
          { id: 's2-lstm-vs-gru', title: 'LSTM vs GRU', difficulty: 'intermediate', readingMinutes: 18, description: 'When to use which and empirical comparisons.', buildsOn: '06-rnns/c3-gru/s1-gru-architecture' },
          { id: 's3-minimal-rnns', title: 'Minimal & Simplified RNNs', difficulty: 'advanced', readingMinutes: 20, description: 'SRU, QRNN, and other efficient recurrent variants.', buildsOn: '06-rnns/c3-gru/s2-lstm-vs-gru' },
        ],
      },
      {
        id: 'c4-bidirectional',
        title: 'Bidirectional & Deep RNNs',
        description: 'Processing sequences in both directions and stacking recurrent layers.',
        difficulty: 'intermediate',
        estimatedMinutes: 200,
        sections: [
          { id: 's1-bidirectional', title: 'Bidirectional RNNs', difficulty: 'intermediate', readingMinutes: 20, description: 'Forward and backward processing for full context.', buildsOn: '06-rnns/c3-gru/s3-minimal-rnns' },
          { id: 's2-deep-rnns', title: 'Stacked & Deep RNNs', difficulty: 'intermediate', readingMinutes: 20, description: 'Multi-layer recurrent networks and residual connections.', buildsOn: '06-rnns/c4-bidirectional/s1-bidirectional' },
          { id: 's3-encoder-decoder', title: 'Encoder-Decoder Framework', difficulty: 'intermediate', readingMinutes: 22, description: 'The general framework for sequence-to-sequence problems.', buildsOn: '06-rnns/c4-bidirectional/s2-deep-rnns' },
        ],
      },
      {
        id: 'c5-seq2seq',
        title: 'Sequence-to-Sequence Models',
        description: 'End-to-end learning for translation, summarization, and more.',
        difficulty: 'intermediate',
        estimatedMinutes: 260,
        sections: [
          { id: 's1-seq2seq-basics', title: 'Seq2Seq Architecture', difficulty: 'intermediate', readingMinutes: 22, description: 'Encoder-decoder with teacher forcing and beam search.', buildsOn: '06-rnns/c4-bidirectional/s3-encoder-decoder' },
          { id: 's2-attention-intro', title: 'Attention for Seq2Seq', difficulty: 'intermediate', readingMinutes: 25, description: 'Bahdanau and Luong attention for alignment.', buildsOn: '06-rnns/c5-seq2seq/s1-seq2seq-basics' },
          { id: 's3-copy-pointer', title: 'Copy & Pointer Networks', difficulty: 'advanced', readingMinutes: 22, description: 'Mechanisms for copying from input sequences.', buildsOn: '06-rnns/c5-seq2seq/s2-attention-intro' },
        ],
      },
    ],
  },
  {
    id: '07-transformers',
    title: 'Transformers & Attention',
    icon: '⊛',
    colorHex: '#3b82f6',
    description: 'The architecture that revolutionized deep learning — self-attention, multi-head attention, and the full Transformer model.',
    prerequisites: ['06-rnns'],
    dlRelevance: 99,
    estimatedHours: 50,
    difficulty: 'intermediate',
    chapters: [
      {
        id: 'c1-attention-mechanism',
        title: 'Attention Mechanism',
        description: 'The fundamental operation of attending to relevant parts of input.',
        difficulty: 'intermediate',
        estimatedMinutes: 260,
        sections: [
          { id: 's1-qkv', title: 'Queries, Keys & Values', difficulty: 'intermediate', readingMinutes: 25, description: 'The QKV framework and scaled dot-product attention.', buildsOn: '06-rnns/c5-seq2seq/s3-copy-pointer' },
          { id: 's2-attention-patterns', title: 'Attention Patterns & Scores', difficulty: 'intermediate', readingMinutes: 22, description: 'How attention distributes weight across positions.', buildsOn: '07-transformers/c1-attention-mechanism/s1-qkv' },
          { id: 's3-attention-variants', title: 'Additive & Multiplicative Attention', difficulty: 'intermediate', readingMinutes: 22, description: 'Different scoring functions for computing attention.', buildsOn: '07-transformers/c1-attention-mechanism/s2-attention-patterns' },
        ],
      },
      {
        id: 'c2-self-attention',
        title: 'Self-Attention & Multi-Head',
        description: 'Attending within a sequence and using multiple attention heads.',
        difficulty: 'intermediate',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-self-attention', title: 'Self-Attention', difficulty: 'intermediate', readingMinutes: 25, description: 'A sequence attending to itself — the key innovation.', buildsOn: '07-transformers/c1-attention-mechanism/s3-attention-variants' },
          { id: 's2-multi-head', title: 'Multi-Head Attention', difficulty: 'intermediate', readingMinutes: 25, description: 'Parallel attention heads for diverse representations.', buildsOn: '07-transformers/c2-self-attention/s1-self-attention' },
          { id: 's3-cross-attention', title: 'Cross-Attention', difficulty: 'intermediate', readingMinutes: 22, description: 'Attending between different sequences — encoder-decoder bridging.', buildsOn: '07-transformers/c2-self-attention/s2-multi-head' },
        ],
      },
      {
        id: 'c3-transformer-architecture',
        title: 'The Transformer Architecture',
        description: 'The complete Transformer with encoder, decoder, and all components.',
        difficulty: 'intermediate',
        estimatedMinutes: 320,
        sections: [
          { id: 's1-encoder', title: 'Transformer Encoder', difficulty: 'intermediate', readingMinutes: 28, description: 'Self-attention, FFN, residuals, and layer norm in the encoder.', buildsOn: '07-transformers/c2-self-attention/s3-cross-attention' },
          { id: 's2-decoder', title: 'Transformer Decoder', difficulty: 'intermediate', readingMinutes: 28, description: 'Masked self-attention, cross-attention, and autoregressive generation.', buildsOn: '07-transformers/c3-transformer-architecture/s1-encoder' },
          { id: 's3-training-inference', title: 'Training & Inference', difficulty: 'intermediate', readingMinutes: 25, description: 'Teacher forcing, KV caching, and efficient inference.', buildsOn: '07-transformers/c3-transformer-architecture/s2-decoder' },
        ],
      },
      {
        id: 'c4-positional-encoding',
        title: 'Positional Encoding',
        description: 'How Transformers encode sequence order without recurrence.',
        difficulty: 'intermediate',
        estimatedMinutes: 240,
        sections: [
          { id: 's1-sinusoidal', title: 'Sinusoidal Encoding', difficulty: 'intermediate', readingMinutes: 22, description: 'The original fixed positional encoding from Vaswani et al.', buildsOn: '07-transformers/c3-transformer-architecture/s3-training-inference' },
          { id: 's2-learned', title: 'Learned Positional Embeddings', difficulty: 'intermediate', readingMinutes: 20, description: 'Trainable position embeddings used in BERT and GPT.', buildsOn: '07-transformers/c4-positional-encoding/s1-sinusoidal' },
          { id: 's3-rope-alibi', title: 'RoPE & ALiBi', difficulty: 'advanced', readingMinutes: 25, description: 'Rotary embeddings and attention with linear biases for length generalization.', buildsOn: '07-transformers/c4-positional-encoding/s2-learned' },
        ],
      },
      {
        id: 'c5-efficient-attention',
        title: 'Efficient Attention Mechanisms',
        description: 'Scaling attention to long sequences and large models.',
        difficulty: 'advanced',
        estimatedMinutes: 280,
        sections: [
          { id: 's1-flash-attention', title: 'Flash Attention', difficulty: 'advanced', readingMinutes: 28, description: 'IO-aware exact attention with tiling for GPU efficiency.', buildsOn: '07-transformers/c4-positional-encoding/s3-rope-alibi' },
          { id: 's2-linear-attention', title: 'Linear & Sparse Attention', difficulty: 'advanced', readingMinutes: 25, description: 'Linearized attention, Longformer, and BigBird approaches.', buildsOn: '07-transformers/c5-efficient-attention/s1-flash-attention' },
          { id: 's3-grouped-query', title: 'Grouped-Query & Multi-Query Attention', difficulty: 'advanced', readingMinutes: 25, description: 'Reducing KV cache with shared keys and values.', buildsOn: '07-transformers/c5-efficient-attention/s2-linear-attention' },
        ],
      },
    ],
  },
  {
    id: '08-nlp',
    title: 'Natural Language Processing',
    icon: '📝',
    colorHex: '#06b6d4',
    description: 'Deep learning for text — word embeddings, language models, BERT, GPT, and modern NLP pipelines.',
    prerequisites: ['07-transformers'],
    dlRelevance: 95,
    estimatedHours: 45,
    difficulty: 'intermediate',
    chapters: [
      { id: 'c1-word-embeddings', title: 'Word Embeddings', description: 'Distributed representations of words in continuous vector spaces.', difficulty: 'intermediate', estimatedMinutes: 260,
        sections: [
          { id: 's1-word2vec', title: 'Word2Vec', difficulty: 'intermediate', readingMinutes: 25, description: 'Skip-gram and CBOW models for learning word vectors.', buildsOn: '07-transformers/c5-efficient-attention/s3-grouped-query' },
          { id: 's2-glove-fasttext', title: 'GloVe & FastText', difficulty: 'intermediate', readingMinutes: 22, description: 'Co-occurrence statistics and subword embeddings.', buildsOn: '08-nlp/c1-word-embeddings/s1-word2vec' },
          { id: 's3-contextual', title: 'Contextual Embeddings', difficulty: 'intermediate', readingMinutes: 25, description: 'ELMo and the shift to context-dependent representations.', buildsOn: '08-nlp/c1-word-embeddings/s2-glove-fasttext' },
        ],
      },
      { id: 'c2-language-models', title: 'Language Models', description: 'Modeling the probability of text sequences.', difficulty: 'intermediate', estimatedMinutes: 260,
        sections: [
          { id: 's1-ngram-neural', title: 'N-gram to Neural LMs', difficulty: 'intermediate', readingMinutes: 22, description: 'Statistical and neural approaches to language modeling.', buildsOn: '08-nlp/c1-word-embeddings/s3-contextual' },
          { id: 's2-perplexity', title: 'Perplexity & Evaluation', difficulty: 'intermediate', readingMinutes: 20, description: 'Measuring language model quality.', buildsOn: '08-nlp/c2-language-models/s1-ngram-neural' },
          { id: 's3-tokenization', title: 'Tokenization: BPE, WordPiece', difficulty: 'intermediate', readingMinutes: 25, description: 'Subword tokenization methods for neural models.', buildsOn: '08-nlp/c2-language-models/s2-perplexity' },
        ],
      },
      { id: 'c3-bert-gpt', title: 'BERT, GPT & Pre-training', description: 'The pre-train/fine-tune paradigm that transformed NLP.', difficulty: 'intermediate', estimatedMinutes: 320,
        sections: [
          { id: 's1-bert', title: 'BERT & Masked LM', difficulty: 'intermediate', readingMinutes: 28, description: 'Bidirectional pre-training with masked language modeling.', buildsOn: '08-nlp/c2-language-models/s3-tokenization' },
          { id: 's2-gpt', title: 'GPT & Autoregressive LM', difficulty: 'intermediate', readingMinutes: 28, description: 'Unidirectional pre-training and in-context learning.', buildsOn: '08-nlp/c3-bert-gpt/s1-bert' },
          { id: 's3-t5-bart', title: 'T5, BART & Seq2Seq Pre-training', difficulty: 'intermediate', readingMinutes: 25, description: 'Encoder-decoder pre-training with text-to-text framing.', buildsOn: '08-nlp/c3-bert-gpt/s2-gpt' },
        ],
      },
      { id: 'c4-text-classification', title: 'Text Classification', description: 'Sentiment analysis, topic classification, and intent detection.', difficulty: 'intermediate', estimatedMinutes: 220,
        sections: [
          { id: 's1-fine-tuning', title: 'Fine-tuning for Classification', difficulty: 'intermediate', readingMinutes: 22, description: 'Adapting pre-trained models for downstream classification.', buildsOn: '08-nlp/c3-bert-gpt/s3-t5-bart' },
          { id: 's2-sentiment', title: 'Sentiment Analysis', difficulty: 'intermediate', readingMinutes: 20, description: 'Polarity detection and aspect-based sentiment.', buildsOn: '08-nlp/c4-text-classification/s1-fine-tuning' },
          { id: 's3-few-shot', title: 'Few-Shot & Zero-Shot Classification', difficulty: 'advanced', readingMinutes: 25, description: 'Classification with minimal labeled data via prompting.', buildsOn: '08-nlp/c4-text-classification/s2-sentiment' },
        ],
      },
      { id: 'c5-ner-qa', title: 'NER, QA & Information Extraction', description: 'Structured extraction of entities, answers, and relations from text.', difficulty: 'intermediate', estimatedMinutes: 260,
        sections: [
          { id: 's1-ner', title: 'Named Entity Recognition', difficulty: 'intermediate', readingMinutes: 22, description: 'Token-level classification for entity extraction.', buildsOn: '08-nlp/c4-text-classification/s3-few-shot' },
          { id: 's2-qa', title: 'Question Answering', difficulty: 'intermediate', readingMinutes: 25, description: 'Extractive and generative QA with Transformers.', buildsOn: '08-nlp/c5-ner-qa/s1-ner' },
          { id: 's3-relation-extraction', title: 'Relation Extraction', difficulty: 'advanced', readingMinutes: 25, description: 'Extracting structured relations between entities.', buildsOn: '08-nlp/c5-ner-qa/s2-qa' },
        ],
      },
      { id: 'c6-machine-translation', title: 'Machine Translation', description: 'Neural approaches to translating between languages.', difficulty: 'intermediate', estimatedMinutes: 240,
        sections: [
          { id: 's1-nmt', title: 'Neural Machine Translation', difficulty: 'intermediate', readingMinutes: 25, description: 'Encoder-decoder and attention-based translation.', buildsOn: '08-nlp/c5-ner-qa/s3-relation-extraction' },
          { id: 's2-multilingual', title: 'Multilingual Models', difficulty: 'intermediate', readingMinutes: 22, description: 'mBERT, XLM-R, and cross-lingual transfer.', buildsOn: '08-nlp/c6-machine-translation/s1-nmt' },
          { id: 's3-evaluation', title: 'MT Evaluation: BLEU, COMET', difficulty: 'intermediate', readingMinutes: 20, description: 'Metrics for translation quality.', buildsOn: '08-nlp/c6-machine-translation/s2-multilingual' },
        ],
      },
    ],
  },
  {
    id: '09-computer-vision',
    title: 'Computer Vision',
    icon: '👁',
    colorHex: '#10b981',
    description: 'Deep learning for images — classification, detection, face recognition, segmentation, pose estimation, and Vision Transformers.',
    prerequisites: ['05-cnns'],
    dlRelevance: 95,
    estimatedHours: 45,
    difficulty: 'intermediate',
    chapters: [
      { id: 'c1-image-classification', title: 'Image Classification', description: 'Categorizing images into predefined classes.', difficulty: 'intermediate', estimatedMinutes: 240,
        sections: [
          { id: 's1-training-pipeline', title: 'Training Pipeline', difficulty: 'intermediate', readingMinutes: 22, description: 'Data loading, augmentation, training, and evaluation for image classifiers.', buildsOn: '05-cnns/c6-semantic-segmentation/s3-panoptic' },
          { id: 's2-transfer-learning', title: 'Transfer Learning', difficulty: 'intermediate', readingMinutes: 22, description: 'Fine-tuning pre-trained models for new domains.', buildsOn: '09-computer-vision/c1-image-classification/s1-training-pipeline' },
          { id: 's3-knowledge-distillation', title: 'Knowledge Distillation', difficulty: 'advanced', readingMinutes: 25, description: 'Training smaller models from larger teacher networks.', buildsOn: '09-computer-vision/c1-image-classification/s2-transfer-learning' },
        ],
      },
      { id: 'c2-object-detection', title: 'Object Detection & Localization', description: 'Finding and localizing objects in images.', difficulty: 'intermediate', estimatedMinutes: 200,
        sections: [
          { id: 's1-anchor-based', title: 'Anchor-Based Methods', difficulty: 'intermediate', readingMinutes: 22, description: 'Anchor boxes, IoU matching, and NMS.', buildsOn: '09-computer-vision/c1-image-classification/s3-knowledge-distillation' },
          { id: 's2-anchor-free', title: 'Anchor-Free Methods', difficulty: 'intermediate', readingMinutes: 22, description: 'CenterNet, FCOS, and keypoint-based detection.', buildsOn: '09-computer-vision/c2-object-detection/s1-anchor-based' },
          { id: 's3-3d-detection', title: '3D Object Detection', difficulty: 'advanced', readingMinutes: 25, description: 'Point clouds, voxels, and BEV-based 3D detection.', buildsOn: '09-computer-vision/c2-object-detection/s2-anchor-free' },
        ],
      },
      { id: 'c3-face-detection', title: 'Face Detection & Recognition', description: 'Detecting faces and verifying/identifying individuals.', difficulty: 'intermediate', estimatedMinutes: 240,
        sections: [
          { id: 's1-face-detection', title: 'Face Detection', difficulty: 'intermediate', readingMinutes: 22, description: 'MTCNN, RetinaFace, and modern face detectors.', buildsOn: '09-computer-vision/c2-object-detection/s3-3d-detection' },
          { id: 's2-face-recognition', title: 'Face Recognition', difficulty: 'intermediate', readingMinutes: 25, description: 'FaceNet, ArcFace, and metric learning for face verification.', buildsOn: '09-computer-vision/c3-face-detection/s1-face-detection' },
          { id: 's3-face-generation', title: 'Face Generation & Editing', difficulty: 'advanced', readingMinutes: 25, description: 'StyleGAN, face swapping, and deepfake detection.', buildsOn: '09-computer-vision/c3-face-detection/s2-face-recognition' },
        ],
      },
      { id: 'c4-image-segmentation', title: 'Image Segmentation', description: 'Pixel-level understanding of images.', difficulty: 'intermediate', estimatedMinutes: 200,
        sections: [
          { id: 's1-semantic', title: 'Semantic Segmentation', difficulty: 'intermediate', readingMinutes: 22, description: 'Per-pixel classification with modern architectures.', buildsOn: '09-computer-vision/c3-face-detection/s3-face-generation' },
          { id: 's2-instance', title: 'Instance Segmentation', difficulty: 'intermediate', readingMinutes: 22, description: 'Distinguishing individual object instances.', buildsOn: '09-computer-vision/c4-image-segmentation/s1-semantic' },
          { id: 's3-sam', title: 'Segment Anything (SAM)', difficulty: 'advanced', readingMinutes: 25, description: 'Foundation models for universal segmentation.', buildsOn: '09-computer-vision/c4-image-segmentation/s2-instance' },
        ],
      },
      { id: 'c5-pose-estimation', title: 'Pose Estimation', description: 'Estimating body keypoints and skeletal structure.', difficulty: 'intermediate', estimatedMinutes: 200,
        sections: [
          { id: 's1-2d-pose', title: '2D Pose Estimation', difficulty: 'intermediate', readingMinutes: 22, description: 'Heatmap-based and regression-based keypoint detection.', buildsOn: '09-computer-vision/c4-image-segmentation/s3-sam' },
          { id: 's2-3d-pose', title: '3D Pose Estimation', difficulty: 'advanced', readingMinutes: 25, description: 'Lifting 2D to 3D and volumetric approaches.', buildsOn: '09-computer-vision/c5-pose-estimation/s1-2d-pose' },
          { id: 's3-hand-body', title: 'Hand & Full-Body Tracking', difficulty: 'advanced', readingMinutes: 22, description: 'Dense pose, hand tracking, and mesh recovery.', buildsOn: '09-computer-vision/c5-pose-estimation/s2-3d-pose' },
        ],
      },
      { id: 'c6-vision-transformers', title: 'Vision Transformers', description: 'Applying Transformers to visual recognition tasks.', difficulty: 'advanced', estimatedMinutes: 280,
        sections: [
          { id: 's1-vit', title: 'ViT: Vision Transformer', difficulty: 'advanced', readingMinutes: 25, description: 'Patch embeddings and pure Transformer for image classification.', buildsOn: '09-computer-vision/c5-pose-estimation/s3-hand-body' },
          { id: 's2-deit-swin', title: 'DeiT & Swin Transformer', difficulty: 'advanced', readingMinutes: 28, description: 'Data-efficient training and hierarchical vision Transformers.', buildsOn: '09-computer-vision/c6-vision-transformers/s1-vit' },
          { id: 's3-detection-transformers', title: 'Detection Transformers', difficulty: 'advanced', readingMinutes: 25, description: 'DETR, Deformable DETR, and Transformer-based detection.', buildsOn: '09-computer-vision/c6-vision-transformers/s2-deit-swin' },
        ],
      },
    ],
  },
  {
    id: '10-audio-speech',
    title: 'Audio & Speech Processing',
    icon: '🔊',
    colorHex: '#f59e0b',
    description: 'Deep learning for audio — speech recognition, text-to-speech, music generation, and speaker verification.',
    prerequisites: ['07-transformers'],
    dlRelevance: 85,
    estimatedHours: 35,
    difficulty: 'intermediate',
    chapters: [
      { id: 'c1-audio-representations', title: 'Audio Representations', description: 'Converting raw audio into features for neural networks.', difficulty: 'intermediate', estimatedMinutes: 220,
        sections: [
          { id: 's1-spectrograms', title: 'Spectrograms & Mel Scale', difficulty: 'intermediate', readingMinutes: 22, description: 'Time-frequency representations and mel-frequency analysis.', buildsOn: '07-transformers/c5-efficient-attention/s3-grouped-query' },
          { id: 's2-mfcc', title: 'MFCCs & Filter Banks', difficulty: 'intermediate', readingMinutes: 22, description: 'Cepstral features and perceptually-motivated filter banks.', buildsOn: '10-audio-speech/c1-audio-representations/s1-spectrograms' },
          { id: 's3-learned-features', title: 'Learned Audio Features', difficulty: 'advanced', readingMinutes: 22, description: 'Raw waveform processing and self-supervised audio features.', buildsOn: '10-audio-speech/c1-audio-representations/s2-mfcc' },
        ],
      },
      { id: 'c2-speech-recognition', title: 'Speech Recognition', description: 'Converting speech to text with deep learning.', difficulty: 'intermediate', estimatedMinutes: 260,
        sections: [
          { id: 's1-ctc', title: 'CTC Loss & Decoding', difficulty: 'intermediate', readingMinutes: 25, description: 'Connectionist temporal classification for sequence alignment.', buildsOn: '10-audio-speech/c1-audio-representations/s3-learned-features' },
          { id: 's2-attention-asr', title: 'Attention-Based ASR', difficulty: 'intermediate', readingMinutes: 25, description: 'Listen-Attend-Spell and Transformer-based ASR.', buildsOn: '10-audio-speech/c2-speech-recognition/s1-ctc' },
          { id: 's3-whisper', title: 'Whisper & Foundation ASR', difficulty: 'advanced', readingMinutes: 25, description: 'Large-scale weakly supervised speech recognition.', buildsOn: '10-audio-speech/c2-speech-recognition/s2-attention-asr' },
        ],
      },
      { id: 'c3-tts', title: 'Text-to-Speech', description: 'Generating natural speech from text.', difficulty: 'intermediate', estimatedMinutes: 240,
        sections: [
          { id: 's1-tacotron', title: 'Tacotron & Vocoders', difficulty: 'intermediate', readingMinutes: 25, description: 'Mel spectrogram prediction and waveform generation.', buildsOn: '10-audio-speech/c2-speech-recognition/s3-whisper' },
          { id: 's2-wavenet', title: 'WaveNet & Autoregressive TTS', difficulty: 'advanced', readingMinutes: 25, description: 'Causal dilated convolutions for raw audio generation.', buildsOn: '10-audio-speech/c3-tts/s1-tacotron' },
          { id: 's3-modern-tts', title: 'Modern TTS: VITS, VALL-E', difficulty: 'advanced', readingMinutes: 25, description: 'End-to-end and codec-based speech synthesis.', buildsOn: '10-audio-speech/c3-tts/s2-wavenet' },
        ],
      },
      { id: 'c4-music-generation', title: 'Music & Audio Generation', description: 'Generating music and sound effects with neural networks.', difficulty: 'advanced', estimatedMinutes: 200,
        sections: [
          { id: 's1-music-models', title: 'Music Generation Models', difficulty: 'advanced', readingMinutes: 25, description: 'MuseNet, Jukebox, and MusicLM.', buildsOn: '10-audio-speech/c3-tts/s3-modern-tts' },
          { id: 's2-audio-diffusion', title: 'Audio Diffusion Models', difficulty: 'advanced', readingMinutes: 22, description: 'AudioLDM and diffusion-based sound generation.', buildsOn: '10-audio-speech/c4-music-generation/s1-music-models' },
          { id: 's3-audio-codecs', title: 'Neural Audio Codecs', difficulty: 'advanced', readingMinutes: 22, description: 'Encodec, SoundStream, and discrete audio tokens.', buildsOn: '10-audio-speech/c4-music-generation/s2-audio-diffusion' },
        ],
      },
      { id: 'c5-speaker-verification', title: 'Speaker Verification', description: 'Identifying and verifying speakers from voice.', difficulty: 'intermediate', estimatedMinutes: 180,
        sections: [
          { id: 's1-speaker-embeddings', title: 'Speaker Embeddings', difficulty: 'intermediate', readingMinutes: 22, description: 'x-vectors, ECAPA-TDNN, and speaker representations.', buildsOn: '10-audio-speech/c4-music-generation/s3-audio-codecs' },
          { id: 's2-verification', title: 'Speaker Verification & ID', difficulty: 'intermediate', readingMinutes: 20, description: 'Verification, identification, and diarization tasks.', buildsOn: '10-audio-speech/c5-speaker-verification/s1-speaker-embeddings' },
          { id: 's3-voice-conversion', title: 'Voice Conversion', difficulty: 'advanced', readingMinutes: 22, description: 'Converting one voice to another while preserving content.', buildsOn: '10-audio-speech/c5-speaker-verification/s2-verification' },
        ],
      },
    ],
  },
  {
    id: '11-video-understanding',
    title: 'Video Understanding',
    icon: '🎬',
    colorHex: '#ef4444',
    description: 'Deep learning for video — temporal models, video Transformers, action recognition, and video generation.',
    prerequisites: ['09-computer-vision'],
    dlRelevance: 80,
    estimatedHours: 30,
    difficulty: 'advanced',
    chapters: [
      { id: 'c1-temporal-models', title: 'Temporal Models', description: '3D CNNs and temporal processing for video.', difficulty: 'advanced', estimatedMinutes: 240,
        sections: [
          { id: 's1-3d-cnns', title: '3D CNNs: C3D, I3D', difficulty: 'advanced', readingMinutes: 25, description: '3D convolutions for spatiotemporal feature learning.', buildsOn: '09-computer-vision/c6-vision-transformers/s3-detection-transformers' },
          { id: 's2-slowfast', title: 'SlowFast Networks', difficulty: 'advanced', readingMinutes: 25, description: 'Dual-pathway processing at different temporal rates.', buildsOn: '11-video-understanding/c1-temporal-models/s1-3d-cnns' },
          { id: 's3-temporal-shift', title: 'Temporal Shift & TSM', difficulty: 'advanced', readingMinutes: 22, description: 'Efficient temporal modeling without 3D convolutions.', buildsOn: '11-video-understanding/c1-temporal-models/s2-slowfast' },
        ],
      },
      { id: 'c2-video-transformers', title: 'Video Transformers', description: 'Applying Transformers to video understanding.', difficulty: 'advanced', estimatedMinutes: 240,
        sections: [
          { id: 's1-timesformer', title: 'TimeSformer', difficulty: 'advanced', readingMinutes: 25, description: 'Divided space-time attention for video.', buildsOn: '11-video-understanding/c1-temporal-models/s3-temporal-shift' },
          { id: 's2-vivit', title: 'ViViT & VideoMAE', difficulty: 'advanced', readingMinutes: 25, description: 'Video Vision Transformers and masked video pre-training.', buildsOn: '11-video-understanding/c2-video-transformers/s1-timesformer' },
          { id: 's3-video-llm', title: 'Video-Language Models', difficulty: 'research', readingMinutes: 25, description: 'VideoCLIP, Video-LLaVA, and multimodal video understanding.', buildsOn: '11-video-understanding/c2-video-transformers/s2-vivit' },
        ],
      },
      { id: 'c3-action-recognition', title: 'Action Recognition', description: 'Classifying human actions in video.', difficulty: 'advanced', estimatedMinutes: 200,
        sections: [
          { id: 's1-action-classification', title: 'Action Classification', difficulty: 'advanced', readingMinutes: 22, description: 'Whole-video action classification approaches.', buildsOn: '11-video-understanding/c2-video-transformers/s3-video-llm' },
          { id: 's2-temporal-detection', title: 'Temporal Action Detection', difficulty: 'advanced', readingMinutes: 22, description: 'Localizing actions in time within untrimmed video.', buildsOn: '11-video-understanding/c3-action-recognition/s1-action-classification' },
          { id: 's3-skeleton-based', title: 'Skeleton-Based Recognition', difficulty: 'advanced', readingMinutes: 22, description: 'GCN-based action recognition from skeletal data.', buildsOn: '11-video-understanding/c3-action-recognition/s2-temporal-detection' },
        ],
      },
      { id: 'c4-video-generation', title: 'Video Generation', description: 'Generating and predicting video with deep models.', difficulty: 'research', estimatedMinutes: 240,
        sections: [
          { id: 's1-video-prediction', title: 'Video Prediction', difficulty: 'advanced', readingMinutes: 25, description: 'Predicting future frames from past observations.', buildsOn: '11-video-understanding/c3-action-recognition/s3-skeleton-based' },
          { id: 's2-video-diffusion', title: 'Video Diffusion Models', difficulty: 'research', readingMinutes: 28, description: 'Sora, Runway, and diffusion-based video generation.', buildsOn: '11-video-understanding/c4-video-generation/s1-video-prediction' },
          { id: 's3-video-editing', title: 'Video Editing & Manipulation', difficulty: 'research', readingMinutes: 25, description: 'Neural video editing, inpainting, and style transfer.', buildsOn: '11-video-understanding/c4-video-generation/s2-video-diffusion' },
        ],
      },
    ],
  },
  {
    id: '12-time-series',
    title: 'Time Series & Forecasting',
    icon: '📈',
    colorHex: '#84cc16',
    description: 'Deep learning for temporal data — forecasting, anomaly detection, and time series classification.',
    prerequisites: ['06-rnns', '07-transformers'],
    dlRelevance: 88,
    estimatedHours: 35,
    difficulty: 'intermediate',
    chapters: [
      { id: 'c1-ts-foundations', title: 'Time Series Foundations', description: 'Core concepts for temporal data processing.', difficulty: 'intermediate', estimatedMinutes: 220,
        sections: [
          { id: 's1-ts-concepts', title: 'Time Series Concepts', difficulty: 'intermediate', readingMinutes: 22, description: 'Stationarity, seasonality, trend decomposition.', buildsOn: '06-rnns/c5-seq2seq/s3-copy-pointer' },
          { id: 's2-windowing', title: 'Windowing & Feature Engineering', difficulty: 'intermediate', readingMinutes: 20, description: 'Sliding windows, lag features, and data preparation.', buildsOn: '12-time-series/c1-ts-foundations/s1-ts-concepts' },
          { id: 's3-evaluation', title: 'Forecasting Evaluation', difficulty: 'intermediate', readingMinutes: 20, description: 'MAE, RMSE, MAPE, and proper evaluation protocols.', buildsOn: '12-time-series/c1-ts-foundations/s2-windowing' },
        ],
      },
      { id: 'c2-dl-forecasting', title: 'DL for Forecasting', description: 'Deep learning architectures for time series prediction.', difficulty: 'intermediate', estimatedMinutes: 280,
        sections: [
          { id: 's1-deepar', title: 'DeepAR & Probabilistic Forecasting', difficulty: 'intermediate', readingMinutes: 25, description: 'Autoregressive RNN-based probabilistic forecasting.', buildsOn: '12-time-series/c1-ts-foundations/s3-evaluation' },
          { id: 's2-nbeats', title: 'N-BEATS & N-HiTS', difficulty: 'intermediate', readingMinutes: 25, description: 'Pure DL architectures with basis expansion.', buildsOn: '12-time-series/c2-dl-forecasting/s1-deepar' },
          { id: 's3-tcn', title: 'Temporal CNNs (TCN)', difficulty: 'intermediate', readingMinutes: 22, description: 'Causal dilated convolutions for sequence modeling.', buildsOn: '12-time-series/c2-dl-forecasting/s2-nbeats' },
        ],
      },
      { id: 'c3-temporal-transformers', title: 'Temporal Transformers', description: 'Transformer-based models for time series.', difficulty: 'advanced', estimatedMinutes: 260,
        sections: [
          { id: 's1-informer', title: 'Informer & Autoformer', difficulty: 'advanced', readingMinutes: 25, description: 'Efficient Transformers for long-sequence forecasting.', buildsOn: '12-time-series/c2-dl-forecasting/s3-tcn' },
          { id: 's2-patchtst', title: 'PatchTST & Channel Independence', difficulty: 'advanced', readingMinutes: 25, description: 'Patching time series for Transformer inputs.', buildsOn: '12-time-series/c3-temporal-transformers/s1-informer' },
          { id: 's3-foundation-ts', title: 'Foundation Models for Time Series', difficulty: 'research', readingMinutes: 28, description: 'TimeGPT, Chronos, and pre-trained temporal models.', buildsOn: '12-time-series/c3-temporal-transformers/s2-patchtst' },
        ],
      },
      { id: 'c4-anomaly-detection', title: 'Anomaly Detection', description: 'Detecting unusual patterns in temporal data.', difficulty: 'intermediate', estimatedMinutes: 200,
        sections: [
          { id: 's1-reconstruction', title: 'Reconstruction-Based Detection', difficulty: 'intermediate', readingMinutes: 22, description: 'Autoencoders and reconstruction error for anomalies.', buildsOn: '12-time-series/c3-temporal-transformers/s3-foundation-ts' },
          { id: 's2-forecasting-based', title: 'Forecasting-Based Detection', difficulty: 'intermediate', readingMinutes: 20, description: 'Prediction errors as anomaly signals.', buildsOn: '12-time-series/c4-anomaly-detection/s1-reconstruction' },
          { id: 's3-transformer-anomaly', title: 'Transformer-Based Anomaly Detection', difficulty: 'advanced', readingMinutes: 25, description: 'Anomaly Transformer and attention-based detection.', buildsOn: '12-time-series/c4-anomaly-detection/s2-forecasting-based' },
        ],
      },
      { id: 'c5-ts-classification', title: 'Time Series Classification', description: 'Classifying temporal patterns and sequences.', difficulty: 'intermediate', estimatedMinutes: 200,
        sections: [
          { id: 's1-ts-classification', title: 'TS Classification Approaches', difficulty: 'intermediate', readingMinutes: 22, description: 'CNN, RNN, and Transformer-based classifiers.', buildsOn: '12-time-series/c4-anomaly-detection/s3-transformer-anomaly' },
          { id: 's2-dtw-shapelet', title: 'DTW & Shapelet Features', difficulty: 'intermediate', readingMinutes: 22, description: 'Dynamic time warping and learned shapelets.', buildsOn: '12-time-series/c5-ts-classification/s1-ts-classification' },
          { id: 's3-multivariate', title: 'Multivariate Classification', difficulty: 'advanced', readingMinutes: 22, description: 'Handling multiple correlated time series.', buildsOn: '12-time-series/c5-ts-classification/s2-dtw-shapelet' },
        ],
      },
    ],
  },
  {
    id: '13-generative-models',
    title: 'Generative Models',
    icon: '✦',
    colorHex: '#d946ef',
    description: 'Learning to generate data — autoencoders, VAEs, GANs, normalizing flows, diffusion models, and flow matching.',
    prerequisites: ['04-optimization', '03-regularization'],
    dlRelevance: 95,
    estimatedHours: 50,
    difficulty: 'advanced',
    chapters: [
      { id: 'c1-autoencoders', title: 'Autoencoders & VAEs', description: 'Learning compressed representations and generating data.', difficulty: 'intermediate', estimatedMinutes: 300,
        sections: [
          { id: 's1-autoencoder', title: 'Autoencoders', difficulty: 'intermediate', readingMinutes: 22, description: 'Encoder-decoder for representation learning and reconstruction.', buildsOn: '04-optimization/c5-initialization/s3-fixup-rezero' },
          { id: 's2-vae', title: 'Variational Autoencoders', difficulty: 'advanced', readingMinutes: 28, description: 'ELBO, reparameterization trick, and latent space generation.', buildsOn: '13-generative-models/c1-autoencoders/s1-autoencoder' },
          { id: 's3-vae-variants', title: 'VAE Variants: Beta-VAE, VQ-VAE', difficulty: 'advanced', readingMinutes: 25, description: 'Disentangled and discrete latent representations.', buildsOn: '13-generative-models/c1-autoencoders/s2-vae' },
        ],
      },
      { id: 'c2-gans', title: 'Generative Adversarial Networks', description: 'Two-player minimax games for generation.', difficulty: 'advanced', estimatedMinutes: 320,
        sections: [
          { id: 's1-gan-basics', title: 'GAN Framework', difficulty: 'advanced', readingMinutes: 28, description: 'Generator, discriminator, and the minimax objective.', buildsOn: '13-generative-models/c1-autoencoders/s3-vae-variants' },
          { id: 's2-dcgan-wgan', title: 'DCGAN & WGAN', difficulty: 'advanced', readingMinutes: 25, description: 'Convolutional GANs and Wasserstein distance training.', buildsOn: '13-generative-models/c2-gans/s1-gan-basics' },
          { id: 's3-stylegan', title: 'StyleGAN & Progressive Growing', difficulty: 'advanced', readingMinutes: 28, description: 'High-resolution image synthesis with style-based generation.', buildsOn: '13-generative-models/c2-gans/s2-dcgan-wgan' },
        ],
      },
      { id: 'c3-normalizing-flows', title: 'Normalizing Flows', description: 'Exact likelihood through invertible transformations.', difficulty: 'advanced', estimatedMinutes: 240,
        sections: [
          { id: 's1-flow-basics', title: 'Flow Basics & Change of Variables', difficulty: 'advanced', readingMinutes: 25, description: 'Invertible maps and exact log-likelihood computation.', buildsOn: '13-generative-models/c2-gans/s3-stylegan' },
          { id: 's2-coupling-flows', title: 'Coupling & Autoregressive Flows', difficulty: 'advanced', readingMinutes: 25, description: 'RealNVP, Glow, and masked autoregressive flows.', buildsOn: '13-generative-models/c3-normalizing-flows/s1-flow-basics' },
          { id: 's3-continuous-flows', title: 'Continuous Normalizing Flows', difficulty: 'research', readingMinutes: 28, description: 'Neural ODEs as continuous-time flows.', buildsOn: '13-generative-models/c3-normalizing-flows/s2-coupling-flows' },
        ],
      },
      { id: 'c4-diffusion-models', title: 'Diffusion Models', description: 'Denoising score matching and iterative refinement.', difficulty: 'advanced', estimatedMinutes: 350,
        sections: [
          { id: 's1-ddpm', title: 'DDPM: Denoising Diffusion', difficulty: 'advanced', readingMinutes: 30, description: 'Forward noising process and reverse denoising.', buildsOn: '13-generative-models/c3-normalizing-flows/s3-continuous-flows' },
          { id: 's2-score-matching', title: 'Score Matching & SDEs', difficulty: 'advanced', readingMinutes: 30, description: 'Score-based models and stochastic differential equations.', buildsOn: '13-generative-models/c4-diffusion-models/s1-ddpm' },
          { id: 's3-cfg-guidance', title: 'Classifier-Free Guidance', difficulty: 'advanced', readingMinutes: 25, description: 'Conditional generation through guidance mechanisms.', buildsOn: '13-generative-models/c4-diffusion-models/s2-score-matching' },
        ],
      },
      { id: 'c5-flow-matching', title: 'Flow Matching', description: 'Modern simulation-free training of continuous flows.', difficulty: 'research', estimatedMinutes: 260,
        sections: [
          { id: 's1-optimal-transport', title: 'Optimal Transport Paths', difficulty: 'research', readingMinutes: 28, description: 'OT-based flow matching for straight trajectories.', buildsOn: '13-generative-models/c4-diffusion-models/s3-cfg-guidance' },
          { id: 's2-rectified-flows', title: 'Rectified Flows', difficulty: 'research', readingMinutes: 28, description: 'Straightening flows for faster sampling.', buildsOn: '13-generative-models/c5-flow-matching/s1-optimal-transport' },
          { id: 's3-consistency-models', title: 'Consistency Models', difficulty: 'research', readingMinutes: 25, description: 'Single-step generation through consistency training.', buildsOn: '13-generative-models/c5-flow-matching/s2-rectified-flows' },
        ],
      },
    ],
  },
  {
    id: '14-self-supervised',
    title: 'Self-Supervised & Contrastive Learning',
    icon: '⟡',
    colorHex: '#0ea5e9',
    description: 'Learning without labels — contrastive learning, masked modeling, and knowledge distillation for visual and multimodal representations.',
    prerequisites: ['07-transformers', '09-computer-vision'],
    dlRelevance: 92,
    estimatedHours: 35,
    difficulty: 'advanced',
    chapters: [
      { id: 'c1-pretext-tasks', title: 'Pretext Tasks', description: 'Self-supervised objectives for representation learning.', difficulty: 'advanced', estimatedMinutes: 220,
        sections: [
          { id: 's1-pretext-overview', title: 'Pretext Task Design', difficulty: 'advanced', readingMinutes: 22, description: 'Rotation prediction, jigsaw, colorization, and other pretext tasks.', buildsOn: '09-computer-vision/c6-vision-transformers/s3-detection-transformers' },
          { id: 's2-predictive-learning', title: 'Predictive Self-Supervision', difficulty: 'advanced', readingMinutes: 22, description: 'Predicting missing or future inputs as a learning signal.', buildsOn: '14-self-supervised/c1-pretext-tasks/s1-pretext-overview' },
          { id: 's3-collapse-prevention', title: 'Collapse Prevention', difficulty: 'advanced', readingMinutes: 25, description: 'Avoiding representational collapse in self-supervised learning.', buildsOn: '14-self-supervised/c1-pretext-tasks/s2-predictive-learning' },
        ],
      },
      { id: 'c2-contrastive-learning', title: 'Contrastive Learning', description: 'Learning by contrasting positive and negative pairs.', difficulty: 'advanced', estimatedMinutes: 280,
        sections: [
          { id: 's1-simclr', title: 'SimCLR', difficulty: 'advanced', readingMinutes: 25, description: 'Simple contrastive learning with data augmentation.', buildsOn: '14-self-supervised/c1-pretext-tasks/s3-collapse-prevention' },
          { id: 's2-moco', title: 'MoCo & Memory Banks', difficulty: 'advanced', readingMinutes: 25, description: 'Momentum contrast with queue-based negative sampling.', buildsOn: '14-self-supervised/c2-contrastive-learning/s1-simclr' },
          { id: 's3-byol-vicreg', title: 'BYOL, VICReg & Non-Contrastive', difficulty: 'advanced', readingMinutes: 28, description: 'Learning without negative pairs through variance-covariance regularization.', buildsOn: '14-self-supervised/c2-contrastive-learning/s2-moco' },
        ],
      },
      { id: 'c3-masked-modeling', title: 'Masked Modeling', description: 'Self-supervision through reconstruction of masked inputs.', difficulty: 'advanced', estimatedMinutes: 240,
        sections: [
          { id: 's1-mae', title: 'Masked Autoencoders (MAE)', difficulty: 'advanced', readingMinutes: 25, description: 'Masking and reconstructing image patches.', buildsOn: '14-self-supervised/c2-contrastive-learning/s3-byol-vicreg' },
          { id: 's2-beit', title: 'BEiT & Visual Tokens', difficulty: 'advanced', readingMinutes: 25, description: 'Discrete visual tokens and masked image modeling.', buildsOn: '14-self-supervised/c3-masked-modeling/s1-mae' },
          { id: 's3-data2vec', title: 'data2vec & Multimodal Masking', difficulty: 'research', readingMinutes: 25, description: 'Unified masked prediction across modalities.', buildsOn: '14-self-supervised/c3-masked-modeling/s2-beit' },
        ],
      },
      { id: 'c4-knowledge-distillation', title: 'Knowledge Distillation', description: 'Self-distillation and teacher-student frameworks.', difficulty: 'advanced', estimatedMinutes: 220,
        sections: [
          { id: 's1-dino', title: 'DINO & Self-Distillation', difficulty: 'advanced', readingMinutes: 25, description: 'Self-distillation with no labels for visual features.', buildsOn: '14-self-supervised/c3-masked-modeling/s3-data2vec' },
          { id: 's2-dinov2', title: 'DINOv2 & Foundation Features', difficulty: 'advanced', readingMinutes: 25, description: 'Scaling self-supervised learning to foundation-level features.', buildsOn: '14-self-supervised/c4-knowledge-distillation/s1-dino' },
          { id: 's3-feature-alignment', title: 'Feature Alignment & Transfer', difficulty: 'advanced', readingMinutes: 22, description: 'Using self-supervised features for downstream tasks.', buildsOn: '14-self-supervised/c4-knowledge-distillation/s2-dinov2' },
        ],
      },
    ],
  },
  {
    id: '15-reinforcement-learning',
    title: 'Deep Reinforcement Learning',
    icon: '🎮',
    colorHex: '#7c3aed',
    description: 'Learning through interaction — DQN, policy gradients, actor-critic methods, and RLHF for alignment.',
    prerequisites: ['02-backpropagation'],
    dlRelevance: 85,
    estimatedHours: 40,
    difficulty: 'advanced',
    chapters: [
      { id: 'c1-mdp-basics', title: 'MDP & Value Functions', description: 'Markov decision processes and the foundations of RL.', difficulty: 'intermediate', estimatedMinutes: 260,
        sections: [
          { id: 's1-mdp', title: 'Markov Decision Processes', difficulty: 'intermediate', readingMinutes: 25, description: 'States, actions, rewards, transitions, and policies.', buildsOn: '02-backpropagation/c4-optimization/s3-training-loop' },
          { id: 's2-bellman', title: 'Bellman Equations', difficulty: 'intermediate', readingMinutes: 25, description: 'Value functions and the recursive structure of optimal control.', buildsOn: '15-reinforcement-learning/c1-mdp-basics/s1-mdp' },
          { id: 's3-dynamic-programming', title: 'Dynamic Programming', difficulty: 'intermediate', readingMinutes: 25, description: 'Value iteration and policy iteration algorithms.', buildsOn: '15-reinforcement-learning/c1-mdp-basics/s2-bellman' },
        ],
      },
      { id: 'c2-dqn', title: 'Deep Q-Networks', description: 'Combining deep learning with Q-learning.', difficulty: 'advanced', estimatedMinutes: 260,
        sections: [
          { id: 's1-q-learning', title: 'Q-Learning', difficulty: 'intermediate', readingMinutes: 22, description: 'Tabular Q-learning and temporal difference updates.', buildsOn: '15-reinforcement-learning/c1-mdp-basics/s3-dynamic-programming' },
          { id: 's2-dqn', title: 'DQN Architecture', difficulty: 'advanced', readingMinutes: 28, description: 'Experience replay, target networks, and the DQN breakthrough.', buildsOn: '15-reinforcement-learning/c2-dqn/s1-q-learning' },
          { id: 's3-dqn-improvements', title: 'Double DQN, Dueling, Rainbow', difficulty: 'advanced', readingMinutes: 25, description: 'Improvements to the basic DQN framework.', buildsOn: '15-reinforcement-learning/c2-dqn/s2-dqn' },
        ],
      },
      { id: 'c3-policy-gradient', title: 'Policy Gradient Methods', description: 'Directly optimizing parameterized policies.', difficulty: 'advanced', estimatedMinutes: 240,
        sections: [
          { id: 's1-reinforce', title: 'REINFORCE', difficulty: 'advanced', readingMinutes: 25, description: 'Monte Carlo policy gradient and variance reduction.', buildsOn: '15-reinforcement-learning/c2-dqn/s3-dqn-improvements' },
          { id: 's2-baseline', title: 'Baselines & Advantage', difficulty: 'advanced', readingMinutes: 22, description: 'Advantage functions and variance reduction techniques.', buildsOn: '15-reinforcement-learning/c3-policy-gradient/s1-reinforce' },
          { id: 's3-trpo', title: 'TRPO & Natural Policy Gradient', difficulty: 'advanced', readingMinutes: 28, description: 'Trust regions and natural gradient for stable updates.', buildsOn: '15-reinforcement-learning/c3-policy-gradient/s2-baseline' },
        ],
      },
      { id: 'c4-actor-critic', title: 'Actor-Critic: A3C, PPO', description: 'Combining value and policy methods for stability.', difficulty: 'advanced', estimatedMinutes: 260,
        sections: [
          { id: 's1-a2c-a3c', title: 'A2C & A3C', difficulty: 'advanced', readingMinutes: 25, description: 'Advantage actor-critic with parallel environments.', buildsOn: '15-reinforcement-learning/c3-policy-gradient/s3-trpo' },
          { id: 's2-ppo', title: 'PPO', difficulty: 'advanced', readingMinutes: 28, description: 'Proximal Policy Optimization — the practical default.', buildsOn: '15-reinforcement-learning/c4-actor-critic/s1-a2c-a3c' },
          { id: 's3-sac', title: 'SAC & Off-Policy Methods', difficulty: 'advanced', readingMinutes: 25, description: 'Soft Actor-Critic and maximum entropy RL.', buildsOn: '15-reinforcement-learning/c4-actor-critic/s2-ppo' },
        ],
      },
      { id: 'c5-rlhf', title: 'RLHF & Alignment', description: 'Using RL to align language models with human preferences.', difficulty: 'advanced', estimatedMinutes: 260,
        sections: [
          { id: 's1-reward-modeling', title: 'Reward Modeling', difficulty: 'advanced', readingMinutes: 25, description: 'Learning reward functions from human comparisons.', buildsOn: '15-reinforcement-learning/c4-actor-critic/s3-sac' },
          { id: 's2-rlhf-pipeline', title: 'RLHF Pipeline', difficulty: 'advanced', readingMinutes: 28, description: 'Pre-training, reward modeling, and PPO fine-tuning.', buildsOn: '15-reinforcement-learning/c5-rlhf/s1-reward-modeling' },
          { id: 's3-dpo', title: 'DPO & RLHF Alternatives', difficulty: 'research', readingMinutes: 28, description: 'Direct preference optimization and offline alignment methods.', buildsOn: '15-reinforcement-learning/c5-rlhf/s2-rlhf-pipeline' },
        ],
      },
    ],
  },
  {
    id: '16-graph-neural-networks',
    title: 'Graph Neural Networks',
    icon: '◉',
    colorHex: '#f43f5e',
    description: 'Deep learning on graph-structured data — message passing, GCN, GAT, and applications in molecules and social networks.',
    prerequisites: ['07-transformers'],
    dlRelevance: 82,
    estimatedHours: 30,
    difficulty: 'advanced',
    chapters: [
      { id: 'c1-graph-basics', title: 'Graph Representations', description: 'Representing graphs for neural network processing.', difficulty: 'intermediate', estimatedMinutes: 200,
        sections: [
          { id: 's1-adjacency', title: 'Adjacency & Feature Matrices', difficulty: 'intermediate', readingMinutes: 20, description: 'Graph data structures and node/edge features.', buildsOn: '07-transformers/c5-efficient-attention/s3-grouped-query' },
          { id: 's2-spectral-basics', title: 'Spectral Graph Theory Basics', difficulty: 'advanced', readingMinutes: 25, description: 'Graph Laplacian, eigenvalues, and spectral decomposition.', buildsOn: '16-graph-neural-networks/c1-graph-basics/s1-adjacency' },
          { id: 's3-graph-tasks', title: 'Graph Learning Tasks', difficulty: 'intermediate', readingMinutes: 20, description: 'Node, edge, and graph-level prediction tasks.', buildsOn: '16-graph-neural-networks/c1-graph-basics/s2-spectral-basics' },
        ],
      },
      { id: 'c2-message-passing', title: 'Message Passing & GCN', description: 'The message passing framework and graph convolutional networks.', difficulty: 'advanced', estimatedMinutes: 240,
        sections: [
          { id: 's1-mpnn', title: 'Message Passing Framework', difficulty: 'advanced', readingMinutes: 25, description: 'Aggregate-combine paradigm for graph learning.', buildsOn: '16-graph-neural-networks/c1-graph-basics/s3-graph-tasks' },
          { id: 's2-gcn', title: 'Graph Convolutional Networks', difficulty: 'advanced', readingMinutes: 25, description: 'Spectral and spatial graph convolutions.', buildsOn: '16-graph-neural-networks/c2-message-passing/s1-mpnn' },
          { id: 's3-graphsage', title: 'GraphSAGE & Sampling', difficulty: 'advanced', readingMinutes: 22, description: 'Inductive learning with neighborhood sampling.', buildsOn: '16-graph-neural-networks/c2-message-passing/s2-gcn' },
        ],
      },
      { id: 'c3-graph-attention', title: 'Graph Attention: GAT', description: 'Attention mechanisms for graphs.', difficulty: 'advanced', estimatedMinutes: 200,
        sections: [
          { id: 's1-gat', title: 'Graph Attention Networks', difficulty: 'advanced', readingMinutes: 25, description: 'Learned attention weights between neighbors.', buildsOn: '16-graph-neural-networks/c2-message-passing/s3-graphsage' },
          { id: 's2-gatv2', title: 'GATv2 & Dynamic Attention', difficulty: 'advanced', readingMinutes: 22, description: 'Improved attention computation for graphs.', buildsOn: '16-graph-neural-networks/c3-graph-attention/s1-gat' },
          { id: 's3-heterogeneous', title: 'Heterogeneous Graph Networks', difficulty: 'advanced', readingMinutes: 25, description: 'Handling different node and edge types.', buildsOn: '16-graph-neural-networks/c3-graph-attention/s2-gatv2' },
        ],
      },
      { id: 'c4-graph-transformers', title: 'Graph Transformers', description: 'Applying Transformer architectures to graphs.', difficulty: 'research', estimatedMinutes: 220,
        sections: [
          { id: 's1-positional-encoding-graphs', title: 'Positional Encoding for Graphs', difficulty: 'research', readingMinutes: 25, description: 'Laplacian PE, random walk PE, and graph-specific positions.', buildsOn: '16-graph-neural-networks/c3-graph-attention/s3-heterogeneous' },
          { id: 's2-graph-transformer', title: 'Graph Transformer Architecture', difficulty: 'research', readingMinutes: 25, description: 'Full attention on graphs with structural encoding.', buildsOn: '16-graph-neural-networks/c4-graph-transformers/s1-positional-encoding-graphs' },
          { id: 's3-gps', title: 'GPS & Hybrid Models', difficulty: 'research', readingMinutes: 25, description: 'General Powerful Scalable graph transformers.', buildsOn: '16-graph-neural-networks/c4-graph-transformers/s2-graph-transformer' },
        ],
      },
      { id: 'c5-applications', title: 'Applications', description: 'GNNs for molecules, social networks, and recommendation.', difficulty: 'advanced', estimatedMinutes: 220,
        sections: [
          { id: 's1-molecular', title: 'Molecular Property Prediction', difficulty: 'advanced', readingMinutes: 25, description: 'GNNs for drug discovery and molecular design.', buildsOn: '16-graph-neural-networks/c4-graph-transformers/s3-gps' },
          { id: 's2-social-networks', title: 'Social Network Analysis', difficulty: 'advanced', readingMinutes: 22, description: 'Community detection, link prediction, and influence modeling.', buildsOn: '16-graph-neural-networks/c5-applications/s1-molecular' },
          { id: 's3-recommendation', title: 'Recommendation Systems', difficulty: 'advanced', readingMinutes: 25, description: 'GNN-based collaborative filtering and knowledge graphs.', buildsOn: '16-graph-neural-networks/c5-applications/s2-social-networks' },
        ],
      },
    ],
  },
  {
    id: '17-multimodal',
    title: 'Multimodal & Foundation Models',
    icon: '⬡',
    colorHex: '#2563eb',
    description: 'Models that span modalities — CLIP, multimodal Transformers, text-to-image generation, LLMs, agents, and mixture of experts.',
    prerequisites: ['08-nlp', '13-generative-models'],
    dlRelevance: 98,
    estimatedHours: 45,
    difficulty: 'advanced',
    chapters: [
      { id: 'c1-clip-align', title: 'CLIP & Vision-Language Alignment', description: 'Aligning visual and textual representations.', difficulty: 'advanced', estimatedMinutes: 260,
        sections: [
          { id: 's1-clip', title: 'CLIP Architecture', difficulty: 'advanced', readingMinutes: 28, description: 'Contrastive image-text pre-training at scale.', buildsOn: '08-nlp/c6-machine-translation/s3-evaluation' },
          { id: 's2-open-clip', title: 'OpenCLIP & SigLIP', difficulty: 'advanced', readingMinutes: 25, description: 'Open-source CLIP variants and sigmoid loss.', buildsOn: '17-multimodal/c1-clip-align/s1-clip' },
          { id: 's3-zero-shot', title: 'Zero-Shot Classification & Retrieval', difficulty: 'advanced', readingMinutes: 22, description: 'Using CLIP for zero-shot visual recognition.', buildsOn: '17-multimodal/c1-clip-align/s2-open-clip' },
        ],
      },
      { id: 'c2-multimodal-transformers', title: 'Multimodal Transformers', description: 'Architectures that process multiple modalities jointly.', difficulty: 'advanced', estimatedMinutes: 260,
        sections: [
          { id: 's1-flamingo', title: 'Flamingo & Visual Prompting', difficulty: 'advanced', readingMinutes: 25, description: 'Interleaving visual and text tokens for few-shot learning.', buildsOn: '17-multimodal/c1-clip-align/s3-zero-shot' },
          { id: 's2-llava', title: 'LLaVA & Visual Instruction Tuning', difficulty: 'advanced', readingMinutes: 28, description: 'Connecting vision encoders to language models.', buildsOn: '17-multimodal/c2-multimodal-transformers/s1-flamingo' },
          { id: 's3-unified-models', title: 'Unified Multimodal Models', difficulty: 'research', readingMinutes: 25, description: 'Gemini, GPT-4V, and models that natively process any modality.', buildsOn: '17-multimodal/c2-multimodal-transformers/s2-llava' },
        ],
      },
      { id: 'c3-text-to-image', title: 'Text-to-Image Generation', description: 'Generating images from text descriptions.', difficulty: 'advanced', estimatedMinutes: 280,
        sections: [
          { id: 's1-dalle', title: 'DALL-E & Image Tokens', difficulty: 'advanced', readingMinutes: 25, description: 'Autoregressive image generation from text.', buildsOn: '17-multimodal/c2-multimodal-transformers/s3-unified-models' },
          { id: 's2-stable-diffusion', title: 'Stable Diffusion & Latent Diffusion', difficulty: 'advanced', readingMinutes: 30, description: 'Efficient diffusion in latent space with text conditioning.', buildsOn: '17-multimodal/c3-text-to-image/s1-dalle' },
          { id: 's3-controlnet', title: 'ControlNet & Guided Generation', difficulty: 'advanced', readingMinutes: 25, description: 'Adding spatial control to diffusion models.', buildsOn: '17-multimodal/c3-text-to-image/s2-stable-diffusion' },
        ],
      },
      { id: 'c4-llms', title: 'Large Language Models', description: 'Scaling, emergent abilities, and the LLM revolution.', difficulty: 'advanced', estimatedMinutes: 320,
        sections: [
          { id: 's1-scaling', title: 'Scaling & Training LLMs', difficulty: 'advanced', readingMinutes: 30, description: 'Data, compute, parameters — the scaling recipe.', buildsOn: '17-multimodal/c3-text-to-image/s3-controlnet' },
          { id: 's2-emergence', title: 'Emergent Abilities', difficulty: 'advanced', readingMinutes: 28, description: 'In-context learning, chain-of-thought, and capabilities that emerge at scale.', buildsOn: '17-multimodal/c4-llms/s1-scaling' },
          { id: 's3-efficient-inference', title: 'Efficient Inference', difficulty: 'advanced', readingMinutes: 28, description: 'Quantization, pruning, speculative decoding, and KV cache optimization.', buildsOn: '17-multimodal/c4-llms/s2-emergence' },
        ],
      },
      { id: 'c5-agents', title: 'LLM Agents & Tool Use', description: 'Language models as autonomous agents.', difficulty: 'research', estimatedMinutes: 240,
        sections: [
          { id: 's1-tool-use', title: 'Tool Use & Function Calling', difficulty: 'advanced', readingMinutes: 25, description: 'Teaching LLMs to use external tools and APIs.', buildsOn: '17-multimodal/c4-llms/s3-efficient-inference' },
          { id: 's2-reasoning', title: 'Reasoning & Planning', difficulty: 'research', readingMinutes: 28, description: 'Chain-of-thought, tree-of-thought, and agentic reasoning.', buildsOn: '17-multimodal/c5-agents/s1-tool-use' },
          { id: 's3-agent-frameworks', title: 'Agent Frameworks', difficulty: 'research', readingMinutes: 25, description: 'ReAct, AutoGPT, and multi-agent systems.', buildsOn: '17-multimodal/c5-agents/s2-reasoning' },
        ],
      },
      { id: 'c6-moe', title: 'Mixture of Experts', description: 'Sparse activation for efficient scaling.', difficulty: 'research', estimatedMinutes: 220,
        sections: [
          { id: 's1-moe-basics', title: 'MoE Architecture', difficulty: 'advanced', readingMinutes: 25, description: 'Gating networks, expert routing, and sparse computation.', buildsOn: '17-multimodal/c5-agents/s3-agent-frameworks' },
          { id: 's2-switch-transformer', title: 'Switch Transformer & GShard', difficulty: 'research', readingMinutes: 25, description: 'Simplified routing and distributed MoE training.', buildsOn: '17-multimodal/c6-moe/s1-moe-basics' },
          { id: 's3-mixtral', title: 'Mixtral & Modern MoE', difficulty: 'research', readingMinutes: 25, description: 'Open-source MoE models and practical deployment.', buildsOn: '17-multimodal/c6-moe/s2-switch-transformer' },
        ],
      },
    ],
  },
  {
    id: '18-frontiers',
    title: 'Research Frontiers',
    icon: '◆',
    colorHex: '#dc2626',
    description: 'The cutting edge — state space models, scaling laws, mechanistic interpretability, test-time compute, and world models.',
    prerequisites: ['17-multimodal'],
    dlRelevance: 100,
    estimatedHours: 35,
    difficulty: 'research',
    chapters: [
      { id: 'c1-state-space-models', title: 'State Space Models', description: 'Alternatives to attention for long-range sequence modeling.', difficulty: 'research', estimatedMinutes: 280,
        sections: [
          { id: 's1-s4', title: 'S4 & Structured State Spaces', difficulty: 'research', readingMinutes: 30, description: 'Linear state space layers for efficient long-range modeling.', buildsOn: '17-multimodal/c6-moe/s3-mixtral' },
          { id: 's2-mamba', title: 'Mamba & Selective SSMs', difficulty: 'research', readingMinutes: 30, description: 'Input-dependent selection for efficient sequence modeling.', buildsOn: '18-frontiers/c1-state-space-models/s1-s4' },
          { id: 's3-hybrid-architectures', title: 'Hybrid SSM-Attention Models', difficulty: 'research', readingMinutes: 28, description: 'Combining SSMs with attention for the best of both worlds.', buildsOn: '18-frontiers/c1-state-space-models/s2-mamba' },
        ],
      },
      { id: 'c2-neural-scaling', title: 'Neural Scaling Laws', description: 'Predictable power laws governing model performance.', difficulty: 'research', estimatedMinutes: 240,
        sections: [
          { id: 's1-scaling-laws', title: 'Kaplan & Chinchilla Scaling Laws', difficulty: 'research', readingMinutes: 28, description: 'Power laws relating compute, data, and parameters to loss.', buildsOn: '18-frontiers/c1-state-space-models/s3-hybrid-architectures' },
          { id: 's2-compute-optimal', title: 'Compute-Optimal Training', difficulty: 'research', readingMinutes: 25, description: 'Balancing model size and training tokens for fixed compute.', buildsOn: '18-frontiers/c2-neural-scaling/s1-scaling-laws' },
          { id: 's3-data-scaling', title: 'Data Scaling & Quality', difficulty: 'research', readingMinutes: 25, description: 'Data mixture, quality filters, and scaling data-constrained regimes.', buildsOn: '18-frontiers/c2-neural-scaling/s2-compute-optimal' },
        ],
      },
      { id: 'c3-mechanistic-interp', title: 'Mechanistic Interpretability', description: 'Reverse-engineering neural network computations.', difficulty: 'research', estimatedMinutes: 260,
        sections: [
          { id: 's1-circuits', title: 'Circuits & Features', difficulty: 'research', readingMinutes: 28, description: 'Understanding computation through individual circuits and features.', buildsOn: '18-frontiers/c2-neural-scaling/s3-data-scaling' },
          { id: 's2-probing', title: 'Probing & Activation Patching', difficulty: 'research', readingMinutes: 25, description: 'Techniques for understanding internal representations.', buildsOn: '18-frontiers/c3-mechanistic-interp/s1-circuits' },
          { id: 's3-sparse-autoencoders', title: 'Sparse Autoencoders for Interp', difficulty: 'research', readingMinutes: 28, description: 'Dictionary learning for decomposing neural representations.', buildsOn: '18-frontiers/c3-mechanistic-interp/s2-probing' },
        ],
      },
      { id: 'c4-test-time-compute', title: 'Test-Time Compute', description: 'Spending more compute at inference for better answers.', difficulty: 'research', estimatedMinutes: 220,
        sections: [
          { id: 's1-chain-of-thought', title: 'Chain-of-Thought Reasoning', difficulty: 'research', readingMinutes: 25, description: 'Step-by-step reasoning for complex problems.', buildsOn: '18-frontiers/c3-mechanistic-interp/s3-sparse-autoencoders' },
          { id: 's2-search-verify', title: 'Search & Verification', difficulty: 'research', readingMinutes: 25, description: 'Best-of-N, process reward models, and MCTS for LLMs.', buildsOn: '18-frontiers/c4-test-time-compute/s1-chain-of-thought' },
          { id: 's3-inference-scaling', title: 'Inference Scaling Laws', difficulty: 'research', readingMinutes: 25, description: 'Trading compute for accuracy at inference time.', buildsOn: '18-frontiers/c4-test-time-compute/s2-search-verify' },
        ],
      },
      { id: 'c5-world-models', title: 'World Models & Embodied AI', description: 'Learning models of the world for planning and action.', difficulty: 'research', estimatedMinutes: 240,
        sections: [
          { id: 's1-world-models', title: 'World Models', difficulty: 'research', readingMinutes: 28, description: 'Learning environment dynamics for planning and imagination.', buildsOn: '18-frontiers/c4-test-time-compute/s3-inference-scaling' },
          { id: 's2-embodied-ai', title: 'Embodied AI & Robotics', difficulty: 'research', readingMinutes: 28, description: 'Foundation models for robotic manipulation and navigation.', buildsOn: '18-frontiers/c5-world-models/s1-world-models' },
          { id: 's3-open-problems', title: 'Open Problems in Deep Learning', difficulty: 'research', readingMinutes: 25, description: 'Unsolved challenges: generalization, reasoning, safety, and beyond.', buildsOn: '18-frontiers/c5-world-models/s2-embodied-ai' },
        ],
      },
    ],
  },
];

// ── Helper functions ──

export function getCurriculumById(id) {
  return CURRICULUM.find((s) => s.id === id) || null;
}

export function getChapterById(subjectId, chapterId) {
  const subject = getCurriculumById(subjectId);
  if (!subject) return null;
  return subject.chapters.find((c) => c.id === chapterId) || null;
}

export function getSectionById(subjectId, chapterId, sectionId) {
  const chapter = getChapterById(subjectId, chapterId);
  if (!chapter) return null;
  return chapter.sections?.find((s) => s.id === sectionId) || null;
}

export function getSubjectSectionCount(subjectId) {
  const subject = getCurriculumById(subjectId);
  if (!subject) return 0;
  return subject.chapters.reduce((acc, ch) => acc + (ch.sections?.length || 0), 0);
}

export function getAdjacentSections(subjectId, chapterId, sectionId) {
  const flat = [];
  for (const subject of CURRICULUM) {
    for (const ch of subject.chapters) {
      for (const sec of ch.sections || []) {
        flat.push({
          title: sec.title,
          subjectId: subject.id,
          subjectTitle: subject.title,
          chapterId: ch.id,
          sectionId: sec.id,
        });
      }
    }
  }

  const idx = flat.findIndex(
    (s) => s.subjectId === subjectId && s.chapterId === chapterId && s.sectionId === sectionId
  );

  if (idx === -1) return { prev: null, next: null };

  const prev = idx > 0 ? flat[idx - 1] : null;
  const next = idx < flat.length - 1 ? flat[idx + 1] : null;

  if (prev && prev.subjectId !== subjectId) {
    prev.crossesSubject = true;
  }
  if (next && next.subjectId !== subjectId) {
    next.crossesSubject = true;
  }

  return { prev, next };
}

export function resolveBuildsOn(buildsOnPath) {
  if (!buildsOnPath) return null;
  const parts = buildsOnPath.split('/');
  if (parts.length !== 3) return null;

  const [subjId, chapId, secId] = parts;
  const subject = getCurriculumById(subjId);
  if (!subject) return null;
  const chapter = subject.chapters.find((c) => c.id === chapId);
  if (!chapter) return null;
  const section = chapter.sections?.find((s) => s.id === secId);
  if (!section) return null;

  return {
    title: section.title,
    subjectId: subjId,
    subjectTitle: subject.title,
    chapterId: chapId,
    chapterTitle: chapter.title,
    sectionId: secId,
  };
}

export default CURRICULUM;
