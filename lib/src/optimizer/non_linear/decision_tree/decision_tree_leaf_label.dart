import 'package:ml_linalg/vector.dart';

class DecisionTreeLeafLabel {
  DecisionTreeLeafLabel.categorical(this.categoricalValue, {this.probability})
      : numericalValue = null;

  DecisionTreeLeafLabel.numerical(this.numericalValue, {this.probability})
      : categoricalValue = null;

  final Vector categoricalValue;
  final double numericalValue;
  final double probability;
}
