class TimeBasedLearningRateIterator implements Iterator<double> {
  TimeBasedLearningRateIterator({
    required double initialValue,
    required double decay,
    required int limit,
  })  : _currentValue = initialValue,
        _decay = decay,
        _limit = limit;

  final double _decay;
  final int _limit;

  double _currentValue;
  int _iteration = 1;

  @override
  double get current => _currentValue;

  @override
  bool moveNext() {
    _currentValue = _currentValue / (1 + _decay * _iteration);

    return _iteration++ <= _limit;
  }
}
