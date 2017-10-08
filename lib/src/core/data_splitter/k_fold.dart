part of 'package:dart_ml/src/core/interface.dart';

abstract class KFoldSplitter implements Splitter {
  void configure({int numberOfFolds});
}