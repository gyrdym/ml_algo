import 'package:dart_ml/src/data_splitter/interface/splitter.dart';

abstract class KFoldSplitter implements Splitter {
  void configure({int numberOfFolds});
}