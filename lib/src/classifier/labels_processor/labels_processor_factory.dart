import 'package:ml_algo/src/classifier/labels_processor/labels_processor.dart';

abstract class LabelsProcessorFactory {
  LabelsProcessor create(Type dtype);
}
