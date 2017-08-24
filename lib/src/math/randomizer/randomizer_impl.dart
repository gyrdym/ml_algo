part of 'package:dart_ml/src/implementation.dart';

class _RandomizerImpl implements Randomizer {
  final Random _generator;

  _RandomizerImpl({int seed}) : _generator = new Random(seed);

  int getIntegerFromInterval(int start, int end) => _generator.nextInt(end - start) + start;

  double getDoubleFromInterval(double start, double end) => _generator.nextDouble() * (end - start) + start;
}