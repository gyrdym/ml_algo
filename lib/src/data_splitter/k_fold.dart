import 'package:dart_ml/src/data_splitter/base.dart';

abstract class KFoldSplitter implements Splitter {
  void configure({int numberOfFolds});
}