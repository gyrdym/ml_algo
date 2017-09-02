part of 'package:dart_ml/src/implementation.dart';

class _RandomizerImpl implements Randomizer {
  final Random _generator;

  _RandomizerImpl({int seed}) : _generator = new Random(seed);

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  List<int> getIntegerInterval(int lowerBound, int upperBound) {
    if (lowerBound == upperBound) {
      throw new RangeError('Lower bound and upper bound must be different');
    }

    List<int> interval = _normalizeInterval(lowerBound, upperBound);
    int end = getIntegerFromInterval(1, interval.last);
    int start = getIntegerFromInterval(interval.first, end);

    return [start, end];
  }

  ///returns random interval constrained by [lowerBound] (include) and [upperBound] (exclude)
  List<double> getDoubleInterval(double lowerBound, double upperBound) {
    throw new UnimplementedError('Method wasn\'t implemented yet');
  }

  ///returns random integer from interval that starts with [start] (include) and ends with [end] (exclude)
  int getIntegerFromInterval(int start, int end) {
    if (start == end) {
      throw new RangeError('Start and end values must be different');
    }

    return _generator.nextInt(end - start) + start;
  }

  ///returns random double from interval that starts with [start] (include) and ends with [end] (exclude)
  double getDoubleFromInterval(double start, double end) {
    if (start == end) {
      throw new RangeError('Start and end values must be different');
    }

    return _generator.nextDouble() * (end - start) + start;
  }

  List<num> _normalizeInterval(num start, num end) => start > end ? [end, start] : [start, end];
}