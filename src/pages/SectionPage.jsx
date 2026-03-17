import { useParams, Link } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { motion } from 'framer-motion'
import { getCurriculumById, getChapterById, getSectionById, getAdjacentSections, resolveBuildsOn } from '../subjects/index.js'
import DifficultyBadge from '../components/navigation/DifficultyBadge.jsx'
import PrevNextNav from '../components/navigation/PrevNextNav.jsx'
import Breadcrumbs from '../components/layout/Breadcrumbs.jsx'
import useProgress from '../hooks/useProgress.js'

// Registry of all section content components (lazy-loaded).
const CONTENT_REGISTRY = {
  '01-foundations/c1-perceptron/s1-biological-neuron': lazy(() => import('../subjects/01-foundations/c1-perceptron/s1-biological-neuron.jsx')),
  '01-foundations/c1-perceptron/s2-perceptron-algorithm': lazy(() => import('../subjects/01-foundations/c1-perceptron/s2-perceptron-algorithm.jsx')),
  '01-foundations/c1-perceptron/s3-linear-separability': lazy(() => import('../subjects/01-foundations/c1-perceptron/s3-linear-separability.jsx')),
  '01-foundations/c2-activation-functions/s1-sigmoid-tanh': lazy(() => import('../subjects/01-foundations/c2-activation-functions/s1-sigmoid-tanh.jsx')),
  '01-foundations/c2-activation-functions/s2-relu-family': lazy(() => import('../subjects/01-foundations/c2-activation-functions/s2-relu-family.jsx')),
  '01-foundations/c2-activation-functions/s3-modern-activations': lazy(() => import('../subjects/01-foundations/c2-activation-functions/s3-modern-activations.jsx')),
  '01-foundations/c3-loss-functions/s1-regression-losses': lazy(() => import('../subjects/01-foundations/c3-loss-functions/s1-regression-losses.jsx')),
  '01-foundations/c3-loss-functions/s2-classification-losses': lazy(() => import('../subjects/01-foundations/c3-loss-functions/s2-classification-losses.jsx')),
  '01-foundations/c3-loss-functions/s3-advanced-losses': lazy(() => import('../subjects/01-foundations/c3-loss-functions/s3-advanced-losses.jsx')),
  '01-foundations/c4-universal-approximation/s1-uat': lazy(() => import('../subjects/01-foundations/c4-universal-approximation/s1-uat.jsx')),
  '01-foundations/c4-universal-approximation/s2-depth-vs-width': lazy(() => import('../subjects/01-foundations/c4-universal-approximation/s2-depth-vs-width.jsx')),
  '01-foundations/c4-universal-approximation/s3-expressiveness': lazy(() => import('../subjects/01-foundations/c4-universal-approximation/s3-expressiveness.jsx')),
  '02-backpropagation/c1-computational-graphs/s1-forward-pass': lazy(() => import('../subjects/02-backpropagation/c1-computational-graphs/s1-forward-pass.jsx')),
  '02-backpropagation/c1-computational-graphs/s2-backward-pass': lazy(() => import('../subjects/02-backpropagation/c1-computational-graphs/s2-backward-pass.jsx')),
  '02-backpropagation/c1-computational-graphs/s3-graph-operations': lazy(() => import('../subjects/02-backpropagation/c1-computational-graphs/s3-graph-operations.jsx')),
  '02-backpropagation/c2-chain-rule/s1-chain-rule-review': lazy(() => import('../subjects/02-backpropagation/c2-chain-rule/s1-chain-rule-review.jsx')),
  '02-backpropagation/c2-chain-rule/s2-backprop-algorithm': lazy(() => import('../subjects/02-backpropagation/c2-chain-rule/s2-backprop-algorithm.jsx')),
  '02-backpropagation/c2-chain-rule/s3-vanishing-exploding': lazy(() => import('../subjects/02-backpropagation/c2-chain-rule/s3-vanishing-exploding.jsx')),
  '02-backpropagation/c3-autodiff/s1-forward-mode': lazy(() => import('../subjects/02-backpropagation/c3-autodiff/s1-forward-mode.jsx')),
  '02-backpropagation/c3-autodiff/s2-reverse-mode': lazy(() => import('../subjects/02-backpropagation/c3-autodiff/s2-reverse-mode.jsx')),
  '02-backpropagation/c3-autodiff/s3-higher-order': lazy(() => import('../subjects/02-backpropagation/c3-autodiff/s3-higher-order.jsx')),
  '02-backpropagation/c4-optimization/s1-gradient-descent': lazy(() => import('../subjects/02-backpropagation/c4-optimization/s1-gradient-descent.jsx')),
  '02-backpropagation/c4-optimization/s2-loss-landscapes': lazy(() => import('../subjects/02-backpropagation/c4-optimization/s2-loss-landscapes.jsx')),
  '02-backpropagation/c4-optimization/s3-training-loop': lazy(() => import('../subjects/02-backpropagation/c4-optimization/s3-training-loop.jsx')),
  '03-regularization/c1-overfitting/s1-bias-variance': lazy(() => import('../subjects/03-regularization/c1-overfitting/s1-bias-variance.jsx')),
  '03-regularization/c1-overfitting/s2-overfitting-detection': lazy(() => import('../subjects/03-regularization/c1-overfitting/s2-overfitting-detection.jsx')),
  '03-regularization/c1-overfitting/s3-double-descent': lazy(() => import('../subjects/03-regularization/c1-overfitting/s3-double-descent.jsx')),
  '03-regularization/c2-weight-regularization/s1-l1-l2': lazy(() => import('../subjects/03-regularization/c2-weight-regularization/s1-l1-l2.jsx')),
  '03-regularization/c2-weight-regularization/s2-weight-decay': lazy(() => import('../subjects/03-regularization/c2-weight-regularization/s2-weight-decay.jsx')),
  '03-regularization/c2-weight-regularization/s3-spectral-norm': lazy(() => import('../subjects/03-regularization/c2-weight-regularization/s3-spectral-norm.jsx')),
  '03-regularization/c3-dropout/s1-dropout': lazy(() => import('../subjects/03-regularization/c3-dropout/s1-dropout.jsx')),
  '03-regularization/c3-dropout/s2-dropconnect': lazy(() => import('../subjects/03-regularization/c3-dropout/s2-dropconnect.jsx')),
  '03-regularization/c3-dropout/s3-label-smoothing': lazy(() => import('../subjects/03-regularization/c3-dropout/s3-label-smoothing.jsx')),
  '03-regularization/c4-early-stopping/s1-early-stopping': lazy(() => import('../subjects/03-regularization/c4-early-stopping/s1-early-stopping.jsx')),
  '03-regularization/c4-early-stopping/s2-cross-validation': lazy(() => import('../subjects/03-regularization/c4-early-stopping/s2-cross-validation.jsx')),
  '03-regularization/c4-early-stopping/s3-hyperparameter-tuning': lazy(() => import('../subjects/03-regularization/c4-early-stopping/s3-hyperparameter-tuning.jsx')),
  '03-regularization/c5-data-augmentation/s1-image-augmentation': lazy(() => import('../subjects/03-regularization/c5-data-augmentation/s1-image-augmentation.jsx')),
  '03-regularization/c5-data-augmentation/s2-advanced-augmentation': lazy(() => import('../subjects/03-regularization/c5-data-augmentation/s2-advanced-augmentation.jsx')),
  '03-regularization/c5-data-augmentation/s3-text-audio-augmentation': lazy(() => import('../subjects/03-regularization/c5-data-augmentation/s3-text-audio-augmentation.jsx')),
  '04-optimization/c1-sgd-variants/s1-momentum': lazy(() => import('../subjects/04-optimization/c1-sgd-variants/s1-momentum.jsx')),
  '04-optimization/c1-sgd-variants/s2-nesterov': lazy(() => import('../subjects/04-optimization/c1-sgd-variants/s2-nesterov.jsx')),
  '04-optimization/c1-sgd-variants/s3-convergence': lazy(() => import('../subjects/04-optimization/c1-sgd-variants/s3-convergence.jsx')),
  '04-optimization/c2-adaptive-methods/s1-adagrad': lazy(() => import('../subjects/04-optimization/c2-adaptive-methods/s1-adagrad.jsx')),
  '04-optimization/c2-adaptive-methods/s2-adam': lazy(() => import('../subjects/04-optimization/c2-adaptive-methods/s2-adam.jsx')),
  '04-optimization/c2-adaptive-methods/s3-lion-sophia': lazy(() => import('../subjects/04-optimization/c2-adaptive-methods/s3-lion-sophia.jsx')),
  '04-optimization/c3-learning-rate/s1-warmup-decay': lazy(() => import('../subjects/04-optimization/c3-learning-rate/s1-warmup-decay.jsx')),
  '04-optimization/c3-learning-rate/s2-cosine-annealing': lazy(() => import('../subjects/04-optimization/c3-learning-rate/s2-cosine-annealing.jsx')),
  '04-optimization/c3-learning-rate/s3-cyclic-lr': lazy(() => import('../subjects/04-optimization/c3-learning-rate/s3-cyclic-lr.jsx')),
  '04-optimization/c4-batch-normalization/s1-batch-norm': lazy(() => import('../subjects/04-optimization/c4-batch-normalization/s1-batch-norm.jsx')),
  '04-optimization/c4-batch-normalization/s2-layer-norm': lazy(() => import('../subjects/04-optimization/c4-batch-normalization/s2-layer-norm.jsx')),
  '04-optimization/c4-batch-normalization/s3-rmsnorm-group': lazy(() => import('../subjects/04-optimization/c4-batch-normalization/s3-rmsnorm-group.jsx')),
  '04-optimization/c5-initialization/s1-xavier-he': lazy(() => import('../subjects/04-optimization/c5-initialization/s1-xavier-he.jsx')),
  '04-optimization/c5-initialization/s2-orthogonal': lazy(() => import('../subjects/04-optimization/c5-initialization/s2-orthogonal.jsx')),
  '04-optimization/c5-initialization/s3-fixup-rezero': lazy(() => import('../subjects/04-optimization/c5-initialization/s3-fixup-rezero.jsx')),
  '05-cnns/c1-convolution-operation/s1-discrete-convolution': lazy(() => import('../subjects/05-cnns/c1-convolution-operation/s1-discrete-convolution.jsx')),
  '05-cnns/c1-convolution-operation/s2-stride-dilation': lazy(() => import('../subjects/05-cnns/c1-convolution-operation/s2-stride-dilation.jsx')),
  '05-cnns/c1-convolution-operation/s3-depthwise-separable': lazy(() => import('../subjects/05-cnns/c1-convolution-operation/s3-depthwise-separable.jsx')),
  '05-cnns/c2-pooling-padding/s1-pooling': lazy(() => import('../subjects/05-cnns/c2-pooling-padding/s1-pooling.jsx')),
  '05-cnns/c2-pooling-padding/s2-global-pooling': lazy(() => import('../subjects/05-cnns/c2-pooling-padding/s2-global-pooling.jsx')),
  '05-cnns/c2-pooling-padding/s3-receptive-field': lazy(() => import('../subjects/05-cnns/c2-pooling-padding/s3-receptive-field.jsx')),
  '05-cnns/c3-classic-architectures/s1-lenet-alexnet': lazy(() => import('../subjects/05-cnns/c3-classic-architectures/s1-lenet-alexnet.jsx')),
  '05-cnns/c3-classic-architectures/s2-vgg-inception': lazy(() => import('../subjects/05-cnns/c3-classic-architectures/s2-vgg-inception.jsx')),
  '05-cnns/c3-classic-architectures/s3-resnet': lazy(() => import('../subjects/05-cnns/c3-classic-architectures/s3-resnet.jsx')),
  '05-cnns/c4-modern-architectures/s1-efficientnet': lazy(() => import('../subjects/05-cnns/c4-modern-architectures/s1-efficientnet.jsx')),
  '05-cnns/c4-modern-architectures/s2-convnext': lazy(() => import('../subjects/05-cnns/c4-modern-architectures/s2-convnext.jsx')),
  '05-cnns/c4-modern-architectures/s3-nas': lazy(() => import('../subjects/05-cnns/c4-modern-architectures/s3-nas.jsx')),
  '05-cnns/c5-object-detection/s1-two-stage': lazy(() => import('../subjects/05-cnns/c5-object-detection/s1-two-stage.jsx')),
  '05-cnns/c5-object-detection/s2-one-stage': lazy(() => import('../subjects/05-cnns/c5-object-detection/s2-one-stage.jsx')),
  '05-cnns/c5-object-detection/s3-detr': lazy(() => import('../subjects/05-cnns/c5-object-detection/s3-detr.jsx')),
  '05-cnns/c6-semantic-segmentation/s1-fcn-unet': lazy(() => import('../subjects/05-cnns/c6-semantic-segmentation/s1-fcn-unet.jsx')),
  '05-cnns/c6-semantic-segmentation/s2-deeplab': lazy(() => import('../subjects/05-cnns/c6-semantic-segmentation/s2-deeplab.jsx')),
  '05-cnns/c6-semantic-segmentation/s3-panoptic': lazy(() => import('../subjects/05-cnns/c6-semantic-segmentation/s3-panoptic.jsx')),
  '06-rnns/c1-vanilla-rnn/s1-rnn-basics': lazy(() => import('../subjects/06-rnns/c1-vanilla-rnn/s1-rnn-basics.jsx')),
  '06-rnns/c1-vanilla-rnn/s2-bptt': lazy(() => import('../subjects/06-rnns/c1-vanilla-rnn/s2-bptt.jsx')),
  '06-rnns/c1-vanilla-rnn/s3-rnn-applications': lazy(() => import('../subjects/06-rnns/c1-vanilla-rnn/s3-rnn-applications.jsx')),
  '06-rnns/c2-lstm/s1-lstm-gates': lazy(() => import('../subjects/06-rnns/c2-lstm/s1-lstm-gates.jsx')),
  '06-rnns/c2-lstm/s2-lstm-variants': lazy(() => import('../subjects/06-rnns/c2-lstm/s2-lstm-variants.jsx')),
  '06-rnns/c2-lstm/s3-lstm-training': lazy(() => import('../subjects/06-rnns/c2-lstm/s3-lstm-training.jsx')),
  '06-rnns/c3-gru/s1-gru-architecture': lazy(() => import('../subjects/06-rnns/c3-gru/s1-gru-architecture.jsx')),
  '06-rnns/c3-gru/s2-lstm-vs-gru': lazy(() => import('../subjects/06-rnns/c3-gru/s2-lstm-vs-gru.jsx')),
  '06-rnns/c3-gru/s3-minimal-rnns': lazy(() => import('../subjects/06-rnns/c3-gru/s3-minimal-rnns.jsx')),
  '06-rnns/c4-bidirectional/s1-bidirectional': lazy(() => import('../subjects/06-rnns/c4-bidirectional/s1-bidirectional.jsx')),
  '06-rnns/c4-bidirectional/s2-deep-rnns': lazy(() => import('../subjects/06-rnns/c4-bidirectional/s2-deep-rnns.jsx')),
  '06-rnns/c4-bidirectional/s3-encoder-decoder': lazy(() => import('../subjects/06-rnns/c4-bidirectional/s3-encoder-decoder.jsx')),
  '06-rnns/c5-seq2seq/s1-seq2seq-basics': lazy(() => import('../subjects/06-rnns/c5-seq2seq/s1-seq2seq-basics.jsx')),
  '06-rnns/c5-seq2seq/s2-attention-intro': lazy(() => import('../subjects/06-rnns/c5-seq2seq/s2-attention-intro.jsx')),
  '06-rnns/c5-seq2seq/s3-copy-pointer': lazy(() => import('../subjects/06-rnns/c5-seq2seq/s3-copy-pointer.jsx')),
  '07-transformers/c1-attention-mechanism/s1-qkv': lazy(() => import('../subjects/07-transformers/c1-attention-mechanism/s1-qkv.jsx')),
  '07-transformers/c1-attention-mechanism/s2-attention-patterns': lazy(() => import('../subjects/07-transformers/c1-attention-mechanism/s2-attention-patterns.jsx')),
  '07-transformers/c1-attention-mechanism/s3-attention-variants': lazy(() => import('../subjects/07-transformers/c1-attention-mechanism/s3-attention-variants.jsx')),
  '07-transformers/c2-self-attention/s1-self-attention': lazy(() => import('../subjects/07-transformers/c2-self-attention/s1-self-attention.jsx')),
  '07-transformers/c2-self-attention/s2-multi-head': lazy(() => import('../subjects/07-transformers/c2-self-attention/s2-multi-head.jsx')),
  '07-transformers/c2-self-attention/s3-cross-attention': lazy(() => import('../subjects/07-transformers/c2-self-attention/s3-cross-attention.jsx')),
  '07-transformers/c3-transformer-architecture/s1-encoder': lazy(() => import('../subjects/07-transformers/c3-transformer-architecture/s1-encoder.jsx')),
  '07-transformers/c3-transformer-architecture/s2-decoder': lazy(() => import('../subjects/07-transformers/c3-transformer-architecture/s2-decoder.jsx')),
  '07-transformers/c3-transformer-architecture/s3-training-inference': lazy(() => import('../subjects/07-transformers/c3-transformer-architecture/s3-training-inference.jsx')),
  '07-transformers/c4-positional-encoding/s1-sinusoidal': lazy(() => import('../subjects/07-transformers/c4-positional-encoding/s1-sinusoidal.jsx')),
  '07-transformers/c4-positional-encoding/s2-learned': lazy(() => import('../subjects/07-transformers/c4-positional-encoding/s2-learned.jsx')),
  '07-transformers/c4-positional-encoding/s3-rope-alibi': lazy(() => import('../subjects/07-transformers/c4-positional-encoding/s3-rope-alibi.jsx')),
  '07-transformers/c5-efficient-attention/s1-flash-attention': lazy(() => import('../subjects/07-transformers/c5-efficient-attention/s1-flash-attention.jsx')),
  '07-transformers/c5-efficient-attention/s2-linear-attention': lazy(() => import('../subjects/07-transformers/c5-efficient-attention/s2-linear-attention.jsx')),
  '07-transformers/c5-efficient-attention/s3-grouped-query': lazy(() => import('../subjects/07-transformers/c5-efficient-attention/s3-grouped-query.jsx')),
  '08-nlp/c1-word-embeddings/s1-word2vec': lazy(() => import('../subjects/08-nlp/c1-word-embeddings/s1-word2vec.jsx')),
  '08-nlp/c1-word-embeddings/s2-glove-fasttext': lazy(() => import('../subjects/08-nlp/c1-word-embeddings/s2-glove-fasttext.jsx')),
  '08-nlp/c1-word-embeddings/s3-contextual': lazy(() => import('../subjects/08-nlp/c1-word-embeddings/s3-contextual.jsx')),
  '08-nlp/c2-language-models/s1-ngram-neural': lazy(() => import('../subjects/08-nlp/c2-language-models/s1-ngram-neural.jsx')),
  '08-nlp/c2-language-models/s2-perplexity': lazy(() => import('../subjects/08-nlp/c2-language-models/s2-perplexity.jsx')),
  '08-nlp/c2-language-models/s3-tokenization': lazy(() => import('../subjects/08-nlp/c2-language-models/s3-tokenization.jsx')),
  '08-nlp/c3-bert-gpt/s1-bert': lazy(() => import('../subjects/08-nlp/c3-bert-gpt/s1-bert.jsx')),
  '08-nlp/c3-bert-gpt/s2-gpt': lazy(() => import('../subjects/08-nlp/c3-bert-gpt/s2-gpt.jsx')),
  '08-nlp/c3-bert-gpt/s3-t5-bart': lazy(() => import('../subjects/08-nlp/c3-bert-gpt/s3-t5-bart.jsx')),
  '08-nlp/c4-text-classification/s1-fine-tuning': lazy(() => import('../subjects/08-nlp/c4-text-classification/s1-fine-tuning.jsx')),
  '08-nlp/c4-text-classification/s2-sentiment': lazy(() => import('../subjects/08-nlp/c4-text-classification/s2-sentiment.jsx')),
  '08-nlp/c4-text-classification/s3-few-shot': lazy(() => import('../subjects/08-nlp/c4-text-classification/s3-few-shot.jsx')),
  '08-nlp/c5-ner-qa/s1-ner': lazy(() => import('../subjects/08-nlp/c5-ner-qa/s1-ner.jsx')),
  '08-nlp/c5-ner-qa/s2-qa': lazy(() => import('../subjects/08-nlp/c5-ner-qa/s2-qa.jsx')),
  '08-nlp/c5-ner-qa/s3-relation-extraction': lazy(() => import('../subjects/08-nlp/c5-ner-qa/s3-relation-extraction.jsx')),
  '08-nlp/c6-machine-translation/s1-nmt': lazy(() => import('../subjects/08-nlp/c6-machine-translation/s1-nmt.jsx')),
  '08-nlp/c6-machine-translation/s2-multilingual': lazy(() => import('../subjects/08-nlp/c6-machine-translation/s2-multilingual.jsx')),
  '08-nlp/c6-machine-translation/s3-evaluation': lazy(() => import('../subjects/08-nlp/c6-machine-translation/s3-evaluation.jsx')),
  '09-computer-vision/c1-image-classification/s1-training-pipeline': lazy(() => import('../subjects/09-computer-vision/c1-image-classification/s1-training-pipeline.jsx')),
  '09-computer-vision/c1-image-classification/s2-transfer-learning': lazy(() => import('../subjects/09-computer-vision/c1-image-classification/s2-transfer-learning.jsx')),
  '09-computer-vision/c1-image-classification/s3-knowledge-distillation': lazy(() => import('../subjects/09-computer-vision/c1-image-classification/s3-knowledge-distillation.jsx')),
  '09-computer-vision/c2-object-detection/s1-anchor-based': lazy(() => import('../subjects/09-computer-vision/c2-object-detection/s1-anchor-based.jsx')),
  '09-computer-vision/c2-object-detection/s2-anchor-free': lazy(() => import('../subjects/09-computer-vision/c2-object-detection/s2-anchor-free.jsx')),
  '09-computer-vision/c2-object-detection/s3-3d-detection': lazy(() => import('../subjects/09-computer-vision/c2-object-detection/s3-3d-detection.jsx')),
  '09-computer-vision/c3-face-detection/s1-face-detection': lazy(() => import('../subjects/09-computer-vision/c3-face-detection/s1-face-detection.jsx')),
  '09-computer-vision/c3-face-detection/s2-face-recognition': lazy(() => import('../subjects/09-computer-vision/c3-face-detection/s2-face-recognition.jsx')),
  '09-computer-vision/c3-face-detection/s3-face-generation': lazy(() => import('../subjects/09-computer-vision/c3-face-detection/s3-face-generation.jsx')),
  '09-computer-vision/c4-image-segmentation/s1-semantic': lazy(() => import('../subjects/09-computer-vision/c4-image-segmentation/s1-semantic.jsx')),
  '09-computer-vision/c4-image-segmentation/s2-instance': lazy(() => import('../subjects/09-computer-vision/c4-image-segmentation/s2-instance.jsx')),
  '09-computer-vision/c4-image-segmentation/s3-sam': lazy(() => import('../subjects/09-computer-vision/c4-image-segmentation/s3-sam.jsx')),
  '09-computer-vision/c5-pose-estimation/s1-2d-pose': lazy(() => import('../subjects/09-computer-vision/c5-pose-estimation/s1-2d-pose.jsx')),
  '09-computer-vision/c5-pose-estimation/s2-3d-pose': lazy(() => import('../subjects/09-computer-vision/c5-pose-estimation/s2-3d-pose.jsx')),
  '09-computer-vision/c5-pose-estimation/s3-hand-body': lazy(() => import('../subjects/09-computer-vision/c5-pose-estimation/s3-hand-body.jsx')),
  '09-computer-vision/c6-vision-transformers/s1-vit': lazy(() => import('../subjects/09-computer-vision/c6-vision-transformers/s1-vit.jsx')),
  '09-computer-vision/c6-vision-transformers/s2-deit-swin': lazy(() => import('../subjects/09-computer-vision/c6-vision-transformers/s2-deit-swin.jsx')),
  '09-computer-vision/c6-vision-transformers/s3-detection-transformers': lazy(() => import('../subjects/09-computer-vision/c6-vision-transformers/s3-detection-transformers.jsx')),
  '10-audio-speech/c1-audio-representations/s1-spectrograms': lazy(() => import('../subjects/10-audio-speech/c1-audio-representations/s1-spectrograms.jsx')),
  '10-audio-speech/c1-audio-representations/s2-mfcc': lazy(() => import('../subjects/10-audio-speech/c1-audio-representations/s2-mfcc.jsx')),
  '10-audio-speech/c1-audio-representations/s3-learned-features': lazy(() => import('../subjects/10-audio-speech/c1-audio-representations/s3-learned-features.jsx')),
  '10-audio-speech/c2-speech-recognition/s1-ctc': lazy(() => import('../subjects/10-audio-speech/c2-speech-recognition/s1-ctc.jsx')),
  '10-audio-speech/c2-speech-recognition/s2-attention-asr': lazy(() => import('../subjects/10-audio-speech/c2-speech-recognition/s2-attention-asr.jsx')),
  '10-audio-speech/c2-speech-recognition/s3-whisper': lazy(() => import('../subjects/10-audio-speech/c2-speech-recognition/s3-whisper.jsx')),
  '10-audio-speech/c3-tts/s1-tacotron': lazy(() => import('../subjects/10-audio-speech/c3-tts/s1-tacotron.jsx')),
  '10-audio-speech/c3-tts/s2-wavenet': lazy(() => import('../subjects/10-audio-speech/c3-tts/s2-wavenet.jsx')),
  '10-audio-speech/c3-tts/s3-modern-tts': lazy(() => import('../subjects/10-audio-speech/c3-tts/s3-modern-tts.jsx')),
  '10-audio-speech/c4-music-generation/s1-music-models': lazy(() => import('../subjects/10-audio-speech/c4-music-generation/s1-music-models.jsx')),
  '10-audio-speech/c4-music-generation/s2-audio-diffusion': lazy(() => import('../subjects/10-audio-speech/c4-music-generation/s2-audio-diffusion.jsx')),
  '10-audio-speech/c4-music-generation/s3-audio-codecs': lazy(() => import('../subjects/10-audio-speech/c4-music-generation/s3-audio-codecs.jsx')),
  '10-audio-speech/c5-speaker-verification/s1-speaker-embeddings': lazy(() => import('../subjects/10-audio-speech/c5-speaker-verification/s1-speaker-embeddings.jsx')),
  '10-audio-speech/c5-speaker-verification/s2-verification': lazy(() => import('../subjects/10-audio-speech/c5-speaker-verification/s2-verification.jsx')),
  '10-audio-speech/c5-speaker-verification/s3-voice-conversion': lazy(() => import('../subjects/10-audio-speech/c5-speaker-verification/s3-voice-conversion.jsx')),
  '11-video-understanding/c1-temporal-models/s1-3d-cnns': lazy(() => import('../subjects/11-video-understanding/c1-temporal-models/s1-3d-cnns.jsx')),
  '11-video-understanding/c1-temporal-models/s2-slowfast': lazy(() => import('../subjects/11-video-understanding/c1-temporal-models/s2-slowfast.jsx')),
  '11-video-understanding/c1-temporal-models/s3-temporal-shift': lazy(() => import('../subjects/11-video-understanding/c1-temporal-models/s3-temporal-shift.jsx')),
  '11-video-understanding/c2-video-transformers/s1-timesformer': lazy(() => import('../subjects/11-video-understanding/c2-video-transformers/s1-timesformer.jsx')),
  '11-video-understanding/c2-video-transformers/s2-vivit': lazy(() => import('../subjects/11-video-understanding/c2-video-transformers/s2-vivit.jsx')),
  '11-video-understanding/c2-video-transformers/s3-video-llm': lazy(() => import('../subjects/11-video-understanding/c2-video-transformers/s3-video-llm.jsx')),
  '11-video-understanding/c3-action-recognition/s1-action-classification': lazy(() => import('../subjects/11-video-understanding/c3-action-recognition/s1-action-classification.jsx')),
  '11-video-understanding/c3-action-recognition/s2-temporal-detection': lazy(() => import('../subjects/11-video-understanding/c3-action-recognition/s2-temporal-detection.jsx')),
  '11-video-understanding/c3-action-recognition/s3-skeleton-based': lazy(() => import('../subjects/11-video-understanding/c3-action-recognition/s3-skeleton-based.jsx')),
  '11-video-understanding/c4-video-generation/s1-video-prediction': lazy(() => import('../subjects/11-video-understanding/c4-video-generation/s1-video-prediction.jsx')),
  '11-video-understanding/c4-video-generation/s2-video-diffusion': lazy(() => import('../subjects/11-video-understanding/c4-video-generation/s2-video-diffusion.jsx')),
  '11-video-understanding/c4-video-generation/s3-video-editing': lazy(() => import('../subjects/11-video-understanding/c4-video-generation/s3-video-editing.jsx')),
  '12-time-series/c1-ts-foundations/s1-ts-concepts': lazy(() => import('../subjects/12-time-series/c1-ts-foundations/s1-ts-concepts.jsx')),
  '12-time-series/c1-ts-foundations/s2-windowing': lazy(() => import('../subjects/12-time-series/c1-ts-foundations/s2-windowing.jsx')),
  '12-time-series/c1-ts-foundations/s3-evaluation': lazy(() => import('../subjects/12-time-series/c1-ts-foundations/s3-evaluation.jsx')),
  '12-time-series/c2-dl-forecasting/s1-deepar': lazy(() => import('../subjects/12-time-series/c2-dl-forecasting/s1-deepar.jsx')),
  '12-time-series/c2-dl-forecasting/s2-nbeats': lazy(() => import('../subjects/12-time-series/c2-dl-forecasting/s2-nbeats.jsx')),
  '12-time-series/c2-dl-forecasting/s3-tcn': lazy(() => import('../subjects/12-time-series/c2-dl-forecasting/s3-tcn.jsx')),
  '12-time-series/c3-temporal-transformers/s1-informer': lazy(() => import('../subjects/12-time-series/c3-temporal-transformers/s1-informer.jsx')),
  '12-time-series/c3-temporal-transformers/s2-patchtst': lazy(() => import('../subjects/12-time-series/c3-temporal-transformers/s2-patchtst.jsx')),
  '12-time-series/c3-temporal-transformers/s3-foundation-ts': lazy(() => import('../subjects/12-time-series/c3-temporal-transformers/s3-foundation-ts.jsx')),
  '12-time-series/c4-anomaly-detection/s1-reconstruction': lazy(() => import('../subjects/12-time-series/c4-anomaly-detection/s1-reconstruction.jsx')),
  '12-time-series/c4-anomaly-detection/s2-forecasting-based': lazy(() => import('../subjects/12-time-series/c4-anomaly-detection/s2-forecasting-based.jsx')),
  '12-time-series/c4-anomaly-detection/s3-transformer-anomaly': lazy(() => import('../subjects/12-time-series/c4-anomaly-detection/s3-transformer-anomaly.jsx')),
  '12-time-series/c5-ts-classification/s1-ts-classification': lazy(() => import('../subjects/12-time-series/c5-ts-classification/s1-ts-classification.jsx')),
  '12-time-series/c5-ts-classification/s2-dtw-shapelet': lazy(() => import('../subjects/12-time-series/c5-ts-classification/s2-dtw-shapelet.jsx')),
  '12-time-series/c5-ts-classification/s3-multivariate': lazy(() => import('../subjects/12-time-series/c5-ts-classification/s3-multivariate.jsx')),
  '13-generative-models/c1-autoencoders/s1-autoencoder': lazy(() => import('../subjects/13-generative-models/c1-autoencoders/s1-autoencoder.jsx')),
  '13-generative-models/c1-autoencoders/s2-vae': lazy(() => import('../subjects/13-generative-models/c1-autoencoders/s2-vae.jsx')),
  '13-generative-models/c1-autoencoders/s3-vae-variants': lazy(() => import('../subjects/13-generative-models/c1-autoencoders/s3-vae-variants.jsx')),
  '13-generative-models/c2-gans/s1-gan-basics': lazy(() => import('../subjects/13-generative-models/c2-gans/s1-gan-basics.jsx')),
  '13-generative-models/c2-gans/s2-dcgan-wgan': lazy(() => import('../subjects/13-generative-models/c2-gans/s2-dcgan-wgan.jsx')),
  '13-generative-models/c2-gans/s3-stylegan': lazy(() => import('../subjects/13-generative-models/c2-gans/s3-stylegan.jsx')),
  '13-generative-models/c3-normalizing-flows/s1-flow-basics': lazy(() => import('../subjects/13-generative-models/c3-normalizing-flows/s1-flow-basics.jsx')),
  '13-generative-models/c3-normalizing-flows/s2-coupling-flows': lazy(() => import('../subjects/13-generative-models/c3-normalizing-flows/s2-coupling-flows.jsx')),
  '13-generative-models/c3-normalizing-flows/s3-continuous-flows': lazy(() => import('../subjects/13-generative-models/c3-normalizing-flows/s3-continuous-flows.jsx')),
  '13-generative-models/c4-diffusion-models/s1-ddpm': lazy(() => import('../subjects/13-generative-models/c4-diffusion-models/s1-ddpm.jsx')),
  '13-generative-models/c4-diffusion-models/s2-score-matching': lazy(() => import('../subjects/13-generative-models/c4-diffusion-models/s2-score-matching.jsx')),
  '13-generative-models/c4-diffusion-models/s3-cfg-guidance': lazy(() => import('../subjects/13-generative-models/c4-diffusion-models/s3-cfg-guidance.jsx')),
  '13-generative-models/c5-flow-matching/s1-optimal-transport': lazy(() => import('../subjects/13-generative-models/c5-flow-matching/s1-optimal-transport.jsx')),
  '13-generative-models/c5-flow-matching/s2-rectified-flows': lazy(() => import('../subjects/13-generative-models/c5-flow-matching/s2-rectified-flows.jsx')),
  '13-generative-models/c5-flow-matching/s3-consistency-models': lazy(() => import('../subjects/13-generative-models/c5-flow-matching/s3-consistency-models.jsx')),
  '14-self-supervised/c1-pretext-tasks/s1-pretext-overview': lazy(() => import('../subjects/14-self-supervised/c1-pretext-tasks/s1-pretext-overview.jsx')),
  '14-self-supervised/c1-pretext-tasks/s2-predictive-learning': lazy(() => import('../subjects/14-self-supervised/c1-pretext-tasks/s2-predictive-learning.jsx')),
  '14-self-supervised/c1-pretext-tasks/s3-collapse-prevention': lazy(() => import('../subjects/14-self-supervised/c1-pretext-tasks/s3-collapse-prevention.jsx')),
  '14-self-supervised/c2-contrastive-learning/s1-simclr': lazy(() => import('../subjects/14-self-supervised/c2-contrastive-learning/s1-simclr.jsx')),
  '14-self-supervised/c2-contrastive-learning/s2-moco': lazy(() => import('../subjects/14-self-supervised/c2-contrastive-learning/s2-moco.jsx')),
  '14-self-supervised/c2-contrastive-learning/s3-byol-vicreg': lazy(() => import('../subjects/14-self-supervised/c2-contrastive-learning/s3-byol-vicreg.jsx')),
  '14-self-supervised/c3-masked-modeling/s1-mae': lazy(() => import('../subjects/14-self-supervised/c3-masked-modeling/s1-mae.jsx')),
  '14-self-supervised/c3-masked-modeling/s2-beit': lazy(() => import('../subjects/14-self-supervised/c3-masked-modeling/s2-beit.jsx')),
  '14-self-supervised/c3-masked-modeling/s3-data2vec': lazy(() => import('../subjects/14-self-supervised/c3-masked-modeling/s3-data2vec.jsx')),
  '14-self-supervised/c4-knowledge-distillation/s1-dino': lazy(() => import('../subjects/14-self-supervised/c4-knowledge-distillation/s1-dino.jsx')),
  '14-self-supervised/c4-knowledge-distillation/s2-dinov2': lazy(() => import('../subjects/14-self-supervised/c4-knowledge-distillation/s2-dinov2.jsx')),
  '14-self-supervised/c4-knowledge-distillation/s3-feature-alignment': lazy(() => import('../subjects/14-self-supervised/c4-knowledge-distillation/s3-feature-alignment.jsx')),
  '15-reinforcement-learning/c1-mdp-basics/s1-mdp': lazy(() => import('../subjects/15-reinforcement-learning/c1-mdp-basics/s1-mdp.jsx')),
  '15-reinforcement-learning/c1-mdp-basics/s2-bellman': lazy(() => import('../subjects/15-reinforcement-learning/c1-mdp-basics/s2-bellman.jsx')),
  '15-reinforcement-learning/c1-mdp-basics/s3-dynamic-programming': lazy(() => import('../subjects/15-reinforcement-learning/c1-mdp-basics/s3-dynamic-programming.jsx')),
  '15-reinforcement-learning/c2-dqn/s1-q-learning': lazy(() => import('../subjects/15-reinforcement-learning/c2-dqn/s1-q-learning.jsx')),
  '15-reinforcement-learning/c2-dqn/s2-dqn': lazy(() => import('../subjects/15-reinforcement-learning/c2-dqn/s2-dqn.jsx')),
  '15-reinforcement-learning/c2-dqn/s3-dqn-improvements': lazy(() => import('../subjects/15-reinforcement-learning/c2-dqn/s3-dqn-improvements.jsx')),
  '15-reinforcement-learning/c3-policy-gradient/s1-reinforce': lazy(() => import('../subjects/15-reinforcement-learning/c3-policy-gradient/s1-reinforce.jsx')),
  '15-reinforcement-learning/c3-policy-gradient/s2-baseline': lazy(() => import('../subjects/15-reinforcement-learning/c3-policy-gradient/s2-baseline.jsx')),
  '15-reinforcement-learning/c3-policy-gradient/s3-trpo': lazy(() => import('../subjects/15-reinforcement-learning/c3-policy-gradient/s3-trpo.jsx')),
  '15-reinforcement-learning/c4-actor-critic/s1-a2c-a3c': lazy(() => import('../subjects/15-reinforcement-learning/c4-actor-critic/s1-a2c-a3c.jsx')),
  '15-reinforcement-learning/c4-actor-critic/s2-ppo': lazy(() => import('../subjects/15-reinforcement-learning/c4-actor-critic/s2-ppo.jsx')),
  '15-reinforcement-learning/c4-actor-critic/s3-sac': lazy(() => import('../subjects/15-reinforcement-learning/c4-actor-critic/s3-sac.jsx')),
  '15-reinforcement-learning/c5-rlhf/s1-reward-modeling': lazy(() => import('../subjects/15-reinforcement-learning/c5-rlhf/s1-reward-modeling.jsx')),
  '15-reinforcement-learning/c5-rlhf/s2-rlhf-pipeline': lazy(() => import('../subjects/15-reinforcement-learning/c5-rlhf/s2-rlhf-pipeline.jsx')),
  '15-reinforcement-learning/c5-rlhf/s3-dpo': lazy(() => import('../subjects/15-reinforcement-learning/c5-rlhf/s3-dpo.jsx')),
  '16-graph-neural-networks/c1-graph-basics/s1-adjacency': lazy(() => import('../subjects/16-graph-neural-networks/c1-graph-basics/s1-adjacency.jsx')),
  '16-graph-neural-networks/c1-graph-basics/s2-spectral-basics': lazy(() => import('../subjects/16-graph-neural-networks/c1-graph-basics/s2-spectral-basics.jsx')),
  '16-graph-neural-networks/c1-graph-basics/s3-graph-tasks': lazy(() => import('../subjects/16-graph-neural-networks/c1-graph-basics/s3-graph-tasks.jsx')),
  '16-graph-neural-networks/c2-message-passing/s1-mpnn': lazy(() => import('../subjects/16-graph-neural-networks/c2-message-passing/s1-mpnn.jsx')),
  '16-graph-neural-networks/c2-message-passing/s2-gcn': lazy(() => import('../subjects/16-graph-neural-networks/c2-message-passing/s2-gcn.jsx')),
  '16-graph-neural-networks/c2-message-passing/s3-graphsage': lazy(() => import('../subjects/16-graph-neural-networks/c2-message-passing/s3-graphsage.jsx')),
  '16-graph-neural-networks/c3-graph-attention/s1-gat': lazy(() => import('../subjects/16-graph-neural-networks/c3-graph-attention/s1-gat.jsx')),
  '16-graph-neural-networks/c3-graph-attention/s2-gatv2': lazy(() => import('../subjects/16-graph-neural-networks/c3-graph-attention/s2-gatv2.jsx')),
  '16-graph-neural-networks/c3-graph-attention/s3-heterogeneous': lazy(() => import('../subjects/16-graph-neural-networks/c3-graph-attention/s3-heterogeneous.jsx')),
  '16-graph-neural-networks/c4-graph-transformers/s1-positional-encoding-graphs': lazy(() => import('../subjects/16-graph-neural-networks/c4-graph-transformers/s1-positional-encoding-graphs.jsx')),
  '16-graph-neural-networks/c4-graph-transformers/s2-graph-transformer': lazy(() => import('../subjects/16-graph-neural-networks/c4-graph-transformers/s2-graph-transformer.jsx')),
  '16-graph-neural-networks/c4-graph-transformers/s3-gps': lazy(() => import('../subjects/16-graph-neural-networks/c4-graph-transformers/s3-gps.jsx')),
  '16-graph-neural-networks/c5-applications/s1-molecular': lazy(() => import('../subjects/16-graph-neural-networks/c5-applications/s1-molecular.jsx')),
  '16-graph-neural-networks/c5-applications/s2-social-networks': lazy(() => import('../subjects/16-graph-neural-networks/c5-applications/s2-social-networks.jsx')),
  '16-graph-neural-networks/c5-applications/s3-recommendation': lazy(() => import('../subjects/16-graph-neural-networks/c5-applications/s3-recommendation.jsx')),
  '17-multimodal/c1-clip-align/s1-clip': lazy(() => import('../subjects/17-multimodal/c1-clip-align/s1-clip.jsx')),
  '17-multimodal/c1-clip-align/s2-open-clip': lazy(() => import('../subjects/17-multimodal/c1-clip-align/s2-open-clip.jsx')),
  '17-multimodal/c1-clip-align/s3-zero-shot': lazy(() => import('../subjects/17-multimodal/c1-clip-align/s3-zero-shot.jsx')),
  '17-multimodal/c2-multimodal-transformers/s1-flamingo': lazy(() => import('../subjects/17-multimodal/c2-multimodal-transformers/s1-flamingo.jsx')),
  '17-multimodal/c2-multimodal-transformers/s2-llava': lazy(() => import('../subjects/17-multimodal/c2-multimodal-transformers/s2-llava.jsx')),
  '17-multimodal/c2-multimodal-transformers/s3-unified-models': lazy(() => import('../subjects/17-multimodal/c2-multimodal-transformers/s3-unified-models.jsx')),
  '17-multimodal/c3-text-to-image/s1-dalle': lazy(() => import('../subjects/17-multimodal/c3-text-to-image/s1-dalle.jsx')),
  '17-multimodal/c3-text-to-image/s2-stable-diffusion': lazy(() => import('../subjects/17-multimodal/c3-text-to-image/s2-stable-diffusion.jsx')),
  '17-multimodal/c3-text-to-image/s3-controlnet': lazy(() => import('../subjects/17-multimodal/c3-text-to-image/s3-controlnet.jsx')),
  '17-multimodal/c4-llms/s1-scaling': lazy(() => import('../subjects/17-multimodal/c4-llms/s1-scaling.jsx')),
  '17-multimodal/c4-llms/s2-emergence': lazy(() => import('../subjects/17-multimodal/c4-llms/s2-emergence.jsx')),
  '17-multimodal/c4-llms/s3-efficient-inference': lazy(() => import('../subjects/17-multimodal/c4-llms/s3-efficient-inference.jsx')),
  '17-multimodal/c5-agents/s1-tool-use': lazy(() => import('../subjects/17-multimodal/c5-agents/s1-tool-use.jsx')),
  '17-multimodal/c5-agents/s2-reasoning': lazy(() => import('../subjects/17-multimodal/c5-agents/s2-reasoning.jsx')),
  '17-multimodal/c5-agents/s3-agent-frameworks': lazy(() => import('../subjects/17-multimodal/c5-agents/s3-agent-frameworks.jsx')),
  '17-multimodal/c6-moe/s1-moe-basics': lazy(() => import('../subjects/17-multimodal/c6-moe/s1-moe-basics.jsx')),
  '17-multimodal/c6-moe/s2-switch-transformer': lazy(() => import('../subjects/17-multimodal/c6-moe/s2-switch-transformer.jsx')),
  '17-multimodal/c6-moe/s3-mixtral': lazy(() => import('../subjects/17-multimodal/c6-moe/s3-mixtral.jsx')),
  '18-frontiers/c1-state-space-models/s1-s4': lazy(() => import('../subjects/18-frontiers/c1-state-space-models/s1-s4.jsx')),
  '18-frontiers/c1-state-space-models/s2-mamba': lazy(() => import('../subjects/18-frontiers/c1-state-space-models/s2-mamba.jsx')),
  '18-frontiers/c1-state-space-models/s3-hybrid-architectures': lazy(() => import('../subjects/18-frontiers/c1-state-space-models/s3-hybrid-architectures.jsx')),
  '18-frontiers/c2-neural-scaling/s1-scaling-laws': lazy(() => import('../subjects/18-frontiers/c2-neural-scaling/s1-scaling-laws.jsx')),
  '18-frontiers/c2-neural-scaling/s2-compute-optimal': lazy(() => import('../subjects/18-frontiers/c2-neural-scaling/s2-compute-optimal.jsx')),
  '18-frontiers/c2-neural-scaling/s3-data-scaling': lazy(() => import('../subjects/18-frontiers/c2-neural-scaling/s3-data-scaling.jsx')),
  '18-frontiers/c3-mechanistic-interp/s1-circuits': lazy(() => import('../subjects/18-frontiers/c3-mechanistic-interp/s1-circuits.jsx')),
  '18-frontiers/c3-mechanistic-interp/s2-probing': lazy(() => import('../subjects/18-frontiers/c3-mechanistic-interp/s2-probing.jsx')),
  '18-frontiers/c3-mechanistic-interp/s3-sparse-autoencoders': lazy(() => import('../subjects/18-frontiers/c3-mechanistic-interp/s3-sparse-autoencoders.jsx')),
  '18-frontiers/c4-test-time-compute/s1-chain-of-thought': lazy(() => import('../subjects/18-frontiers/c4-test-time-compute/s1-chain-of-thought.jsx')),
  '18-frontiers/c4-test-time-compute/s2-search-verify': lazy(() => import('../subjects/18-frontiers/c4-test-time-compute/s2-search-verify.jsx')),
  '18-frontiers/c4-test-time-compute/s3-inference-scaling': lazy(() => import('../subjects/18-frontiers/c4-test-time-compute/s3-inference-scaling.jsx')),
  '18-frontiers/c5-world-models/s1-world-models': lazy(() => import('../subjects/18-frontiers/c5-world-models/s1-world-models.jsx')),
  '18-frontiers/c5-world-models/s2-embodied-ai': lazy(() => import('../subjects/18-frontiers/c5-world-models/s2-embodied-ai.jsx')),
  '18-frontiers/c5-world-models/s3-open-problems': lazy(() => import('../subjects/18-frontiers/c5-world-models/s3-open-problems.jsx')),
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function ClockIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  )
}

function BookIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-violet-300 dark:text-violet-700" aria-hidden="true">
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  )
}

function ComingSoonPlaceholder({ section }) {
  return (
    <motion.div
      className="flex flex-col items-center gap-6 rounded-2xl border border-dashed border-violet-200 bg-violet-50/50 px-8 py-16 text-center dark:border-violet-800/40 dark:bg-violet-950/10"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4, delay: 0.1 }}
    >
      <BookIcon />
      <div className="space-y-2">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Content Coming Soon
        </h2>
        <p className="max-w-md text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
          The interactive content for{' '}
          <strong className="font-semibold text-gray-700 dark:text-gray-300">
            {section.title}
          </strong>{' '}
          is being prepared. It will include formal definitions, interactive
          visualizations, mathematical foundations, and PyTorch code examples.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        {['Theory', 'Visualizations', 'Math', 'PyTorch Code'].map((tag) => (
          <span
            key={tag}
            className="rounded-full bg-violet-100 px-3 py-1 text-xs font-medium text-violet-600 dark:bg-violet-900/30 dark:text-violet-400"
          >
            {tag}
          </span>
        ))}
      </div>
    </motion.div>
  )
}

function PrerequisiteBanner({ section, subjectId }) {
  if (!section?.buildsOn) return null
  const prereq = resolveBuildsOn(section.buildsOn)
  if (!prereq) return null

  const isSameSubject = prereq.subjectId === subjectId
  const href = `/subjects/${prereq.subjectId}/${prereq.chapterId}/${prereq.sectionId}`

  return (
    <div className="mb-6 flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50/60 px-4 py-3 dark:border-amber-800/40 dark:bg-amber-950/20">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" aria-hidden="true">
        <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
        <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
      </svg>
      <div className="text-sm leading-relaxed text-amber-900 dark:text-amber-200">
        <span className="font-medium">Builds on: </span>
        <Link
          to={href}
          className="underline decoration-amber-400/60 underline-offset-2 hover:decoration-amber-600 dark:decoration-amber-600/60 dark:hover:decoration-amber-400 transition-colors"
        >
          {prereq.title}
        </Link>
        {!isSameSubject && (
          <span className="ml-1 text-amber-700 dark:text-amber-400/80">
            ({prereq.subjectTitle})
          </span>
        )}
      </div>
    </div>
  )
}

function SectionContent({ subjectId, chapterId, sectionId, section }) {
  const key = `${subjectId}/${chapterId}/${sectionId}`
  const ContentComponent = CONTENT_REGISTRY[key]
  if (ContentComponent) {
    return (
      <Suspense fallback={<div className="py-16 text-center text-gray-400">Loading content…</div>}>
        <ContentComponent />
      </Suspense>
    )
  }
  return <ComingSoonPlaceholder section={section} />
}

export default function SectionPage() {
  const { subjectId, chapterId, sectionId } = useParams()
  const { isComplete, markComplete } = useProgress()

  const subject = getCurriculumById(subjectId)
  const chapter = getChapterById(subjectId, chapterId)
  const section = getSectionById(subjectId, chapterId, sectionId)
  const done = isComplete(subjectId, chapterId, sectionId)

  if (!subject || !chapter || !section) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-6 text-center">
        <div className="text-5xl" aria-hidden="true">∅</div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Section Not Found</h1>
        <p className="text-gray-500 dark:text-gray-400">
          Could not find section &ldquo;{sectionId}&rdquo;.
        </p>
        <Link
          to="/"
          className="rounded-lg bg-violet-600 px-5 py-2 text-sm font-semibold text-white hover:bg-violet-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500"
        >
          Back to Home
        </Link>
      </div>
    )
  }

  const { prev, next } = getAdjacentSections(subjectId, chapterId, sectionId)

  const breadcrumbs = [
    { label: 'Home', href: '/' },
    { label: subject.title, href: `/subjects/${subjectId}` },
    { label: chapter.title, href: `/subjects/${subjectId}/${chapterId}` },
    { label: section.title },
  ]

  function handleMarkComplete() {
    if (!done) {
      markComplete(subjectId, chapterId, sectionId)
    }
  }

  return (
    <div className="min-h-screen">
      <div
        className="relative border-b border-gray-200 dark:border-gray-800"
        style={{ background: `linear-gradient(135deg, ${subject.colorHex}10 0%, transparent 50%)` }}
      >
        <div
          className="absolute left-0 top-0 h-full w-1.5"
          style={{ backgroundColor: subject.colorHex }}
          aria-hidden="true"
        />

        <div className="mx-auto max-w-3xl px-6 py-8 pl-10">
          <Breadcrumbs items={breadcrumbs} />

          <motion.div
            className="mt-4"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
          >
            <h1 className="text-2xl font-extrabold text-gray-900 dark:text-white sm:text-3xl leading-snug">
              {section.title}
            </h1>

            <div className="mt-3 flex flex-wrap items-center gap-3">
              <DifficultyBadge level={section.difficulty} />
              {section.readingMinutes && (
                <span className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400">
                  <ClockIcon />
                  {section.readingMinutes} min read
                </span>
              )}
              {done && (
                <span className="flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">
                  <CheckIcon />
                  Completed
                </span>
              )}
            </div>

            {section.description && (
              <p className="mt-3 text-gray-600 dark:text-gray-400 leading-relaxed">
                {section.description}
              </p>
            )}
          </motion.div>
        </div>
      </div>

      <div className="mx-auto max-w-3xl px-6 py-12">
        <PrerequisiteBanner section={section} subjectId={subjectId} />

        <SectionContent
          subjectId={subjectId}
          chapterId={chapterId}
          sectionId={sectionId}
          section={section}
        />

        <div className="mt-8 flex justify-center">
          <button
            type="button"
            onClick={handleMarkComplete}
            disabled={done}
            className={`inline-flex items-center gap-2 rounded-xl px-6 py-3 text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 focus-visible:ring-offset-2 ${
              done
                ? 'cursor-default bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400'
                : 'bg-violet-600 text-white hover:bg-violet-700 shadow-md hover:shadow-lg'
            }`}
            aria-label={done ? 'Section already marked complete' : 'Mark this section as complete'}
          >
            {done ? (
              <>
                <CheckIcon />
                Marked as Complete
              </>
            ) : (
              'Mark as Complete'
            )}
          </button>
        </div>

        <PrevNextNav prev={prev} next={next} />
      </div>
    </div>
  )
}
