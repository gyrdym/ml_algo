part of 'package:dart_ml/src/predictor/predictor.dart';

class _GradientLinearClassifier<T extends GradientOptimizer> extends _GradientLinearPredictor implements Classifier {
  _GradientLinearClassifier({LossFunction lossFunction, double learningRate, double minWeightsDistance, int iterationLimit, Metric metric,
                              Regularization regularization, ModuleInjector customInjector, alpha})
      : super(metric: metric) {

    injector = customInjector ?? InjectorFactory.create();

    _optimizer = injector.get(T)
      ..init(
        learningRate: learningRate,
        minWeightsDistance: minWeightsDistance,
        iterationLimit: iterationLimit,
        regularization: regularization,
        lossFunction: lossFunction,
        scoreFunction: new ScoreFunction.Linear(),
        alpha: alpha
      );
  }

  @override
  double test(List<Float32x4Vector> features, List<double> origLabels, {Metric metric}) {
    metric = metric ?? this.metric;
    Float32x4Vector prediction = predictClasses(features);
    return metric.getError(prediction, new Float32x4Vector.from(origLabels));
  }

  Float32x4Vector predictClasses(List<Float32x4Vector> features) {
    Float32List probabilities = predict(features).asList();
    return new Float32x4Vector.from(probabilities.map((double value) => value.round() * 1.0));
  }
}
