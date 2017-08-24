part of 'package:dart_ml/src/interface.dart';

abstract class Splitter {
  Iterable<Iterable<int>> split(int numberOfSamples);
}