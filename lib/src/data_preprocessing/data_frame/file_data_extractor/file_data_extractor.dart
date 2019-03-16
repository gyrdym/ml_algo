import 'package:ml_algo/src/data_preprocessing/data_frame/features_extractor/features_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/header_extractor/header_extractor.dart';
import 'package:ml_algo/src/data_preprocessing/data_frame/labels_extractor/labels_extractor.dart';

abstract class FileDataExtractor {
  DataFrameHeaderExtractor get headerExtractor;
  DataFrameFeaturesExtractor get featuresExtractor;
  DataFrameLabelsExtractor get labelsExtractor;


}