import 'package:ml_linalg/vector.dart';

abstract class LabelsProcessor {
  MLVector makeLabelsOneVsAll(MLVector origLabels, double targetLabel);
}
