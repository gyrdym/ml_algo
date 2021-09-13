import 'dart:math' as math;

class StepBasedLearningRateIterator implements Iterator<double> {
  StepBasedLearningRateIterator({
    required double initialValue,
    required double decay,
    required int dropRate,
    required int limit,
  })  : _initialValue = initialValue,
        _currentValue = initialValue,
        _decay = decay,
        _dropRate = dropRate,
        _limit = limit;

  final double _initialValue;
  final double _decay;
  final int _dropRate;
  final int _limit;

  double _currentValue;
  int _iteration = 1;

  @override
  double get current => _currentValue;

  @override
  bool moveNext() {
    _currentValue = _initialValue * math.pow(_decay, ((1 + _iteration) / _dropRate).floor());

    return _iteration++ <= _limit;
  }
}
