import 'dart:math' as math;

class ExponentialLearningRateIterator implements Iterator<double> {
  ExponentialLearningRateIterator({
    required double initialValue,
    required double decay,
    required int limit,
  })  : _initialValue = initialValue,
        _currentValue = initialValue,
        _decay = decay,
        _limit = limit;

  final double _initialValue;
  final double _decay;
  final int _limit;

  double _currentValue;
  int _iteration = 1;

  @override
  double get current => _currentValue;

  @override
  bool moveNext() {
    _currentValue = _initialValue * math.exp(-_decay * _iteration);

    return _iteration++ <= _limit;
  }
}
